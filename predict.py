import argparse
import logging
import numpy as np
import os
import sys
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
    parsing.log_args(args)

    if args.do_predict and os.path.exists(args.result_file):
        logging.info(f"Result file found at {args.result_file}, skipping prediction")

    elif args.do_predict and not os.path.exists(args.result_file):
        # os.makedirs(os.path.join("./results", args.data_name), exist_ok=True)

        # initialization ----------------- model
        assert os.path.exists(args.load_from), f"{args.load_from} does not exist!"
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        state = torch.load(args.load_from)
        pretrain_args = state["args"]
        pretrain_state_dict = state["state_dict"]

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
        model.load_state_dict(pretrain_state_dict)
        logging.info(f"Loaded pretrained state_dict from {args.load_from}")

        model.to(device)
        model.eval()

        logging.info(model)
        logging.info(f"Number of parameters = {param_count(model)}")

        # initialization ----------------- data
        test_dataset = dataset_class(pretrain_args, file=args.test_bin)
        test_dataset.batch(
            batch_type=args.batch_type,
            batch_size=args.predict_batch_size
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda _batch: _batch[0],
            pin_memory=True
        )

        all_predictions = []

        with torch.no_grad():
            for test_idx, test_batch in enumerate(test_loader):
                if test_idx % args.log_iter == 0:
                    logging.info(f"Doing inference on test step {test_idx}")
                    sys.stdout.flush()

                test_batch.to(device)
                results = model.predict_step(
                    reaction_batch=test_batch,
                    batch_size=test_batch.size,
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

        with open(args.result_file, "w") as of:
            of.writelines(all_predictions)

    if args.do_score:
        correct = 0
        invalid = 0

        with open(args.test_tgt, "r") as f:
            total = sum(1 for _ in f)

        accuracies = np.zeros([total, args.n_best], dtype=np.float32)

        with open(args.test_tgt, "r") as f_tgt, open(args.result_file, "r") as f_predict:
            for i, (line_tgt, line_predict) in enumerate(zip(f_tgt, f_predict)):
                smi_tgt = "".join(line_tgt.split())
                smi_tgt = canonicalize_smiles(smi_tgt, trim=False)
                if not smi_tgt or smi_tgt == "CC":
                    continue

                # smi_predict = "".join(line_predict.split())
                line_predict = "".join(line_predict.split())
                smis_predict = line_predict.split(",")
                smis_predict = [canonicalize_smiles(smi, trim=False) for smi in smis_predict]
                if not smis_predict[0]:
                    invalid += 1
                smis_predict = [smi for smi in smis_predict if smi and not smi == "CC"]
                smis_predict = list(dict.fromkeys(smis_predict))

                for j, smi in enumerate(smis_predict):
                    if smi == smi_tgt:
                        accuracies[i, j:] = 1.0
                        break

        logging.info(f"Total: {total}, "
                     f"top 1 invalid: {invalid / total * 100: .2f} %")

        mean_accuracies = np.mean(accuracies, axis=0)
        for n in range(args.n_best):
            logging.info(f"Top {n+1} accuracy: {mean_accuracies[n] * 100: .2f} %")


if __name__ == "__main__":
    predict_parser = get_predict_parser()
    args = predict_parser.parse_args()

    # set random seed (just in case)
    set_seed(args.seed)

    # logger setup
    logger = setup_logger(args, warning_off=True)

    torch.set_printoptions(profile="full")
    main(args)
