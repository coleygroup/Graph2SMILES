import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data_utils import S2SBatch
from onmt.encoders.transformer import TransformerEncoder
from onmt.decoders import TransformerDecoder
from onmt.modules.embeddings import Embeddings
from onmt.translate import BeamSearch, GNMTGlobalScorer, GreedySearch
from typing import Dict


class Seq2Seq(nn.Module):
    def __init__(self, args, vocab: Dict[str, int]):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        while args.enable_amp and not self.vocab_size % 8 == 0:
            self.vocab_size += 1

        self.encoder_embeddings = Embeddings(
            word_vec_size=args.embed_size,
            word_vocab_size=self.vocab_size,
            word_padding_idx=self.vocab["_PAD"],
            position_encoding=True,
            dropout=args.dropout
        )

        self.decoder_embeddings = Embeddings(
            word_vec_size=args.embed_size,
            word_vocab_size=self.vocab_size,
            word_padding_idx=self.vocab["_PAD"],
            position_encoding=True,
            dropout=args.dropout
        )

        if args.share_embeddings:
            self.decoder_embeddings.word_lut.weight = self.encoder_embeddings.word_lut.weight

        self.encoder = TransformerEncoder(
            num_layers=args.decoder_num_layers,
            d_model=args.decoder_hidden_size,
            heads=args.decoder_attn_heads,
            d_ff=args.decoder_filter_size,
            dropout=args.dropout,
            attention_dropout=args.attn_dropout,
            embeddings=self.encoder_embeddings,
            max_relative_positions=0
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
            max_relative_positions=0,
            aan_useffn=False,
            full_context_alignment=False,
            alignment_layer=-3,
            alignment_heads=0
        )

        self.output_layer = nn.Linear(args.decoder_hidden_size, self.vocab_size, bias=True)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab["_PAD"],
            reduction="mean"
        )

    def encode_and_reshape(self, reaction_batch: S2SBatch):
        src = reaction_batch.src_token_ids
        lengths = reaction_batch.src_lengths

        src = src.transpose(0, 1).contiguous().unsqueeze(-1)                # [b, src_t] => [src_t, b, 1]
        emb, out, length = self.encoder(src, lengths)
        self.decoder.init_state(
            src=src,
            memory_bank=out,
            enc_hidden=emb
        )

        return out, length

    def forward(self, reaction_batch: S2SBatch):
        padded_memory_bank, memory_lengths = self.encode_and_reshape(reaction_batch)

        dec_in = reaction_batch.tgt_token_ids[:, :-1]           # pop last and insert SOS for decoder input
        m = nn.ConstantPad1d((1, 0), self.vocab["_SOS"])
        dec_in = m(dec_in)
        dec_in = dec_in.transpose(0, 1).unsqueeze(-1)                           # [b, tgt_t] => [tgt_t, b, 1]

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
        # acc = accs.mean() omg this is so stupid
        acc = accs.sum() / mask.sum()

        return loss, acc

    def predict_step(self, reaction_batch: S2SBatch,
                     batch_size: int, beam_size: int, min_length: int, max_length: int):
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
                n_best=1,           # TODO: this is hard-coded to return top 1 right now
                global_scorer=global_scorer,
                min_length=min_length,
                max_length=max_length,
                return_attention=False,
                block_ngram_repeat=0,
                exclusion_tokens=set(),
                stepwise_penalty=None,
                ratio=0.0
            )

        padded_memory_bank, memory_lengths = self.encode_and_reshape(reaction_batch)

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

            dec_out, dec_attn = self.decoder(
                tgt=decoder_input,
                memory_bank=memory_bank,
                memory_lengths=memory_lengths,
                step=step
            )

            if "std" in dec_attn:
                attn = dec_attn["std"]
            else:
                attn = None

            dec_out = self.output_layer(dec_out)            # [t, b, h] => [t, b, v]
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
