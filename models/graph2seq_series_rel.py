import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.chem_utils import ATOM_FDIM, BOND_FDIM
from utils.data_utils import G2SBatch
from utils.train_utils import log_tensor
from models.attention_xl import AttnEncoderXL
from models.graphfeat import GraphFeatEncoder
from onmt.decoders import TransformerDecoder
from onmt.modules.embeddings import Embeddings
from onmt.translate import BeamSearch, GNMTGlobalScorer, GreedySearch
from typing import Any, Dict


class Graph2SeqSeriesRel(nn.Module):
    def __init__(self, args, vocab: Dict[str, int]):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        while args.enable_amp and not self.vocab_size % 8 == 0:
            self.vocab_size += 1

        self.encoder = GraphFeatEncoder(
            args,
            n_atom_feat=sum(ATOM_FDIM),
            n_bond_feat=BOND_FDIM
        )

        if args.attn_enc_num_layers > 0:
            self.attention_encoder = AttnEncoderXL(args)
        else:
            self.attention_encoder = None

        self.decoder_embeddings = Embeddings(
            word_vec_size=args.embed_size,
            word_vocab_size=self.vocab_size,
            word_padding_idx=self.vocab["_PAD"],
            position_encoding=True,
            dropout=args.dropout
        )

        self.decoder = TransformerDecoder(
            num_layers=args.decoder_num_layers,
            d_model=args.decoder_hidden_size,
            heads=args.decoder_attn_heads,
            d_ff=args.decoder_filter_size,
            copy_attn=False,
            self_attn_type="scaled-dot",
            dropout=args.dropout,
            attention_dropout=args.attn_dropout,
            embeddings=self.decoder_embeddings,
            max_relative_positions=args.max_relative_positions,
            aan_useffn=False,
            full_context_alignment=False,
            alignment_layer=-3,
            alignment_heads=0
        )

        if not args.attn_enc_hidden_size == args.decoder_hidden_size:
            self.bridge_layer = nn.Linear(args.attn_enc_hidden_size, args.decoder_hidden_size, bias=True)

        self.output_layer = nn.Linear(args.decoder_hidden_size, self.vocab_size, bias=True)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab["_PAD"],
            reduction="mean"
        )

    def encode_and_reshape(self, reaction_batch: G2SBatch):
        hatom, _ = self.encoder(reaction_batch)                         # (n_atoms, h)
        if not self.args.attn_enc_hidden_size == self.args.decoder_hidden_size:
            hatom = self.bridge_layer(hatom)                            # bridging

        # hatom reshaping into [t, b, h]
        atom_scope = reaction_batch.atom_scope                          # list of b (n_components, 2)

        memory_lengths = [scope[-1][0] + scope[-1][1] - scope[0][0]
                          for scope in atom_scope]                      # (b, )

        # the 1+ corresponds to Atom(*)
        assert 1 + sum(memory_lengths) == hatom.size(0), \
            f"Memory lengths calculation error, encoder output: {hatom.size(0)}, memory_lengths: {memory_lengths}"

        memory_bank = torch.split(hatom, [1] + memory_lengths, dim=0)   # [n_atoms, h] => 1+b tup of (t, h)
        padded_memory_bank = []
        max_length = max(memory_lengths)

        for length, h in zip(memory_lengths, memory_bank[1:]):
            m = nn.ZeroPad2d((0, 0, 0, max_length - length))
            padded_memory_bank.append(m(h))

        padded_memory_bank = torch.stack(padded_memory_bank, dim=1)     # list of b (max_t, h) => [max_t, b, h]

        memory_lengths = torch.tensor(memory_lengths,
                                      dtype=torch.long,
                                      device=padded_memory_bank.device)

        if self.attention_encoder is not None:
            padded_memory_bank = self.attention_encoder(
                padded_memory_bank,
                memory_lengths,
                reaction_batch.distances
            )

        self.decoder.state["src"] = np.zeros(max_length)    # TODO: this is hardcoded to make transformer decoder work

        return padded_memory_bank, memory_lengths

    def forward(self, reaction_batch: G2SBatch):
        padded_memory_bank, memory_lengths = self.encode_and_reshape(reaction_batch)

        # adapted from onmt.models
        dec_in = reaction_batch.tgt_token_ids[:, :-1]                       # pop last, insert SOS for decoder input
        m = nn.ConstantPad1d((1, 0), self.vocab["_SOS"])
        dec_in = m(dec_in)
        dec_in = dec_in.transpose(0, 1).unsqueeze(-1)                       # [b, max_tgt_t] => [max_tgt_t, b, 1]

        dec_outs, _ = self.decoder(
            tgt=dec_in,
            memory_bank=padded_memory_bank,
            memory_lengths=memory_lengths
        )

        dec_outs = self.output_layer(dec_outs)                                  # [t, b, h] => [t, b, v]
        dec_outs = dec_outs.permute(1, 2, 0)                                    # [t, b, v] => [b, v, t]

        loss = self.criterion(
            input=dec_outs,
            target=reaction_batch.tgt_token_ids
        )

        predictions = torch.argmax(dec_outs, dim=1)                             # [b, t]
        mask = (reaction_batch.tgt_token_ids != self.vocab["_PAD"]).long()
        accs = (predictions == reaction_batch.tgt_token_ids).float()
        accs = accs * mask
        acc = accs.sum() / mask.sum()

        return loss, acc

    def predict_step(self, reaction_batch: G2SBatch,
                     batch_size: int, beam_size: int, n_best: int, temperature: float,
                     min_length: int, max_length: int) -> Dict[str, Any]:
        if beam_size == 1:
            decode_strategy = GreedySearch(
                pad=self.vocab["_PAD"],
                bos=self.vocab["_SOS"],
                eos=self.vocab["_EOS"],
                batch_size=batch_size,
                min_length=min_length,
                max_length=max_length,
                block_ngram_repeat=0,
                exclusion_tokens=set(),
                return_attention=False,
                sampling_temp=0.0,
                keep_topk=1
            )
        else:
            global_scorer = GNMTGlobalScorer(alpha=0.0,
                                             beta=0.0,
                                             length_penalty="none",
                                             coverage_penalty="none")
            decode_strategy = BeamSearch(
                beam_size=beam_size,
                batch_size=batch_size,
                pad=self.vocab["_PAD"],
                bos=self.vocab["_SOS"],
                eos=self.vocab["_EOS"],
                n_best=n_best,
                global_scorer=global_scorer,
                min_length=min_length,
                max_length=max_length,
                return_attention=False,
                block_ngram_repeat=0,
                exclusion_tokens=set(),
                stepwise_penalty=None,
                ratio=0.0
            )

        padded_memory_bank, memory_lengths = self.encode_and_reshape(reaction_batch=reaction_batch)
        # adapted from onmt.translate.translator
        results = {
            "predictions": None,
            "scores": None,
            "attention": None
        }

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = None
        target_prefix = None
        fn_map_state, memory_bank, memory_lengths, src_map = decode_strategy.initialize(
            memory_bank=padded_memory_bank,
            src_lengths=memory_lengths,
            src_map=src_map,
            target_prefix=target_prefix
        )

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(1, -1, 1)

            dec_out, dec_attn = self.decoder(tgt=decoder_input,
                                             memory_bank=memory_bank,
                                             memory_lengths=memory_lengths,
                                             step=step)

            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None

            dec_out = self.output_layer(dec_out)            # [t, b, h] => [t, b, v]
            dec_out = dec_out / temperature
            dec_out = dec_out.squeeze(0)                    # [t, b, v] => [b, v]
            log_probs = F.log_softmax(dec_out, dim=-1)

            # log_probs = self.model.generator(dec_out.squeeze(0))

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices)
                                        for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            if any_finished:
                self.map_state(
                    lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        results["alignment"] = [[] for _ in range(self.args.predict_batch_size)]

        return results

    # adapted from onmt.decoders.transformer
    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        # self.decoder.state["src"] = fn(self.decoder.state["src"], 1)
        # => self.state["src"] = self.state["src"].index_select(1, select_indices)

        if self.decoder.state["cache"] is not None:
            _recursive_map(self.decoder.state["cache"])
