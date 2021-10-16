import argparse
import glob
import logging
import numpy as np
import os
import sys
import time
import torch
from models.graph2seq_series_rel import Graph2SeqSeriesRel
from models.seq2seq import Seq2Seq
from torch.utils.data import DataLoader
from utils import parsing
from utils.data_utils import canonicalize_smiles, load_vocab, S2SDataset, G2SDataset
from utils.train_utils import log_tensor, param_count, set_seed, setup_logger


def get_predict_parser():
    parser = argparse.ArgumentParser("predict")
    parsing.add_common_args(parser)
    parsing.add_preprocess_args(parser)
    parsing.add_train_args(parser)
    parsing.add_predict_args(parser)

    return parser


def main(args):
    start = time.time()
    parsing.log_args(args)

    os.makedirs(os.path.join("./results", args.data_name), exist_ok=True)

    # initialization ----------------- model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    checkpoints = glob.glob(os.path.join(args.load_from, "*.pt"))
    checkpoints = sorted(
        checkpoints,
        key=lambda ckpt: int(ckpt.split(".")[-2].split("_")[-1]),
        reverse=True
    )
    checkpoints = [ckpt for ckpt in checkpoints
                   if (args.checkpoint_step_start <= int(ckpt.split(".")[-2].split("_")[0]))
                   and (args.checkpoint_step_end >= int(ckpt.split(".")[-2].split("_")[0]))]

    model = None
    val_dataset = None
    vocab_tokens = None
    smis_tgt = []
    for ckpt_i, checkpoint in enumerate(checkpoints):
        logging.info(f"Loading from {checkpoint}")
        state = torch.load(checkpoint)

        pretrain_args = state["args"]
        pretrain_state_dict = state["state_dict"]

        if model is None:
            # initialization ----------------- model
            logging.info(f"Model is None, building model")
            logging.info(f"First logging args for training")
            parsing.log_args(pretrain_args)

            for attr in ["mpn_type", "rel_pos"]:
                try:
                    getattr(pretrain_args, attr)
                except AttributeError:
                    setattr(pretrain_args, attr, getattr(args, attr))

            assert args.model == pretrain_args.model, f"Pretrained model is {pretrain_args.model}!"
            if args.model == "s2s":
                model_class = Seq2Seq
                dataset_class = S2SDataset
            elif args.model == "g2s_series_rel":
                model_class = Graph2SeqSeriesRel
                dataset_class = G2SDataset
                args.compute_graph_distance = True
                assert args.compute_graph_distance
            else:
                raise ValueError(f"Model {args.model} not supported!")

            # initialization ----------------- vocab
            vocab = load_vocab(pretrain_args.vocab_file)
            vocab_tokens = [k for k, v in sorted(vocab.items(), key=lambda tup: tup[1])]

            model = model_class(pretrain_args, vocab)
            logging.info(model)
            logging.info(f"Number of parameters = {param_count(model)}")

            # initialization ----------------- data
            val_dataset = dataset_class(pretrain_args, file=args.valid_bin)
            val_dataset.batch(
                batch_type=args.batch_type,
                batch_size=args.predict_batch_size
            )
            with open(args.val_tgt, "r") as f:
                total = sum(1 for _ in f)

            with open(args.val_tgt, "r") as f:
                for line_tgt in f:
                    smi_tgt = "".join(line_tgt.split())
                    smi_tgt = canonicalize_smiles(smi_tgt)
                    smis_tgt.append(smi_tgt)

        model.load_state_dict(pretrain_state_dict)
        logging.info(f"Loaded pretrained state_dict from {checkpoint}")

        model.to(device)
        model.eval()

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=True
        )

        # prediction
        all_predictions = []
        with torch.no_grad():
            for val_idx, val_batch in enumerate(val_loader):
                if val_idx % args.log_iter == 0:
                    logging.info(f"Doing inference on val step {val_idx}, time: {time.time() - start: .2f} s")
                    sys.stdout.flush()

                val_batch.to(device)
                results = model.predict_step(
                    reaction_batch=val_batch,
                    batch_size=val_batch.size,
                    beam_size=args.beam_size,
                    n_best=args.n_best,
                    temperature=args.temperature,
                    min_length=args.predict_min_len,
                    max_length=args.predict_max_len
                )

                for predictions in results["predictions"]:
                    smis = []
                    for prediction in predictions:
                        predicted_idx = prediction.detach().cpu().numpy()
                        predicted_tokens = [vocab_tokens[idx] for idx in predicted_idx[:-1]]
                        smi = " ".join(predicted_tokens)
                        smis.append(smi)
                    smis = ",".join(smis)
                    all_predictions.append(f"{smis}\n")

        # saving prediction results
        result_file = f"{args.result_file}.{ckpt_i}"
        result_stat_file = f"{args.result_file}.stat.{ckpt_i}"
        with open(result_file, "w") as of:
            of.writelines(all_predictions)

        # scoring
        invalid = 0
        accuracies = np.zeros([total, args.n_best], dtype=np.float32)

        with open(result_file, "r") as f_predict:
            for i, (smi_tgt, line_predict) in enumerate(zip(smis_tgt, f_predict)):
                if smi_tgt == "CC":         # problematic SMILES
                    continue

                # smi_predict = "".join(line_predict.split())
                line_predict = "".join(line_predict.split())
                smis_predict = line_predict.split(",")
                smis_predict = [canonicalize_smiles(smi, trim=False, suppress_warning=True) for smi in smis_predict]
                if not smis_predict[0]:
                    invalid += 1
                smis_predict = [smi for smi in smis_predict if smi and not smi == "CC"]

                for j, smi in enumerate(smis_predict):
                    if smi == smi_tgt:
                        accuracies[i, j:] = 1.0
                        break

        with open(result_stat_file, "w") as of:
            line = f"Total: {total}, top 1 invalid: {invalid / total * 100: .2f} %"
            logging.info(line)
            of.write(f"{line}\n")

            mean_accuracies = np.mean(accuracies, axis=0)
            for n in range(args.n_best):
                line = f"Top {n+1} accuracy: {mean_accuracies[n] * 100: .2f} %"
                logging.info(line)
                of.write(f"{line}\n")


if __name__ == "__main__":
    predict_parser = get_predict_parser()
    args = predict_parser.parse_args()

    # set random seed (just in case)
    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args, warning_off=True)

    torch.set_printoptions(profile="full")
    main(args)
