import argparse
import logging


def log_args(args):
    logging.info(f"Logging arguments")
    for k, v in vars(args).items():
        logging.info(f"**** {k} = *{v}*")


def add_common_args(parser):
    group = parser.add_argument_group("Meta")
    group.add_argument("--model", help="Model architecture",
                       choices=["s2s", "g2s", "g2s_series", "g2s_series_rel"], type=str, default="")
    group.add_argument("--data_name", help="Data name", type=str, default="")
    group.add_argument("--task", help="Task", choices=["reaction_prediction", "retrosynthesis", "autoencoding"],
                       type=str, default="")
    group.add_argument("--representation_end", help="Final string representation to be fed",
                       choices=["smiles", "selfies"], type=str, default="")
    group.add_argument("--seed", help="Random seed", type=int, default=42)
    group.add_argument("--max_src_len", help="Max source length", type=int, default=512)
    group.add_argument("--max_tgt_len", help="Max target length", type=int, default=512)
    group.add_argument("--num_workers", help="No. of workers", type=int, default=1)
    group.add_argument("--verbose", help="Whether to enable verbose debugging", action="store_true")

    group = parser.add_argument_group("Paths")
    group.add_argument("--log_file", help="Preprocess log file", type=str, default="")
    group.add_argument("--vocab_file", help="Vocab file", type=str, default="")
    group.add_argument("--preprocess_output_path", help="Path for saving preprocessed outputs",
                       type=str, default="")
    group.add_argument("--save_dir", help="Path for saving checkpoints", type=str, default="")


def add_preprocess_args(parser):
    group = parser.add_argument_group("Preprocessing options")
    # data paths
    group.add_argument("--train_src", help="Train source", type=str, default="")
    group.add_argument("--train_tgt", help="Train target", type=str, default="")
    group.add_argument("--val_src", help="Validation source", type=str, default="")
    group.add_argument("--val_tgt", help="Validation target", type=str, default="")
    group.add_argument("--test_src", help="Test source", type=str, default="")
    group.add_argument("--test_tgt", help="Test target", type=str, default="")
    # options
    group.add_argument("--representation_start", help="Initial string representation to be fed",
                       choices=["smiles"], type=str, default="")
    group.add_argument("--do_tokenize", help="Whether to tokenize the data files", action="store_true")
    group.add_argument("--make_vocab_only", help="Whether to only make vocab", action="store_true")


def add_train_args(parser):
    group = parser.add_argument_group("Training options")
    # file paths
    group.add_argument("--train_bin", help="Train npz", type=str, default="")
    group.add_argument("--valid_bin", help="Valid npz", type=str, default="")
    group.add_argument("--load_from", help="Checkpoint to load", type=str, default="")
    # model params
    group.add_argument("--embed_size", help="Decoder embedding size", type=int, default=256)
    group.add_argument("--share_embeddings", help="Whether to share encoder/decoder embeddings", action="store_true")
    # -------------- mpn encoder ---------------
    group.add_argument("--mpn_type", help="Type of MPN", type=str,
                       choices=["dgcn", "dgat", "dgate", "dgates", "ffn"], default="")
    group.add_argument("--encoder_num_layers", help="No. of layers in transformer/mpn encoder", type=int, default=4)
    group.add_argument("--encoder_hidden_size", help="Encoder hidden size", type=int, default=256)
    group.add_argument("--encoder_attn_heads", help="Encoder no. of attention heads", type=int, default=8)
    group.add_argument("--encoder_filter_size", help="Encoder filter size", type=int, default=2048)
    group.add_argument("--encoder_norm", help="Encoder norm", type=str, default="none")
    group.add_argument("--encoder_skip_connection", help="Encoder skip connection", type=str, default="none")
    group.add_argument("--encoder_positional_encoding", help="Encoder positional encoding", type=str, default="")
    group.add_argument("--encoder_emb_scale", help="How to scale encoder embedding", type=str, default="")
    # -------------- attention encoder ---------------
    group.add_argument("--compute_graph_distance", help="Whether to compute graph distance", action="store_true")
    group.add_argument("--attn_enc_num_layers", help="No. of layers", type=int, default=4)
    group.add_argument("--attn_enc_hidden_size", help="Hidden size", type=int, default=256)
    group.add_argument("--attn_enc_heads", help="Hidden size", type=int, default=8)
    group.add_argument("--attn_enc_filter_size", help="Filter size", type=int, default=2048)
    group.add_argument("--rel_pos", help="type of rel. pos.", type=str, default="none")
    group.add_argument("--rel_pos_buckets", help="No. of relative position buckets", type=int, default=10)
    # -------------- Transformer decoder ---------------
    group.add_argument("--decoder_num_layers", help="No. of layers in transformer decoder", type=int, default=4)
    group.add_argument("--decoder_hidden_size", help="Decoder hidden size", type=int, default=256)
    group.add_argument("--decoder_attn_heads", help="Decoder no. of attention heads", type=int, default=8)
    group.add_argument("--decoder_filter_size", help="Decoder filter size", type=int, default=2048)
    group.add_argument("--dropout", help="Hidden dropout", type=float, default=0.0)
    group.add_argument("--attn_dropout", help="Attention dropout", type=float, default=0.0)
    group.add_argument("--max_relative_positions", help="Max relative positions", type=int, default=0)
    # training params
    group.add_argument("--enable_amp", help="Whether to enable mixed precision training", action="store_true")
    group.add_argument("--epoch", help="Number of training epochs", type=int, default=300)
    group.add_argument("--max_steps", help="Number of max total steps", type=int, default=1000000)
    group.add_argument("--warmup_steps", help="Number of warmup steps", type=int, default=8000)
    group.add_argument("--lr", help="Learning rate", type=float, default=0.0)
    group.add_argument("--beta1", help="Adam beta 1", type=float, default=0.9)
    group.add_argument("--beta2", help="Adam beta 2", type=float, default=0.998)
    group.add_argument("--eps", help="Adam epsilon", type=float, default=1e-9)
    group.add_argument("--weight_decay", help="Adam weight decay", type=float, default=1e-2)
    group.add_argument("--clip_norm", help="Max norm for gradient clipping", type=float, default=20.0)
    group.add_argument("--batch_type", help="batch type", type=str, default="tokens")
    group.add_argument("--train_batch_size", help="Batch size for train", type=int, default=4096)
    group.add_argument("--valid_batch_size", help="Batch size for valid", type=int, default=4096)
    group.add_argument("--accumulation_count", help="No. of batches for gradient accumulation", type=int, default=1)
    group.add_argument("--log_iter", help="No. of steps per logging", type=int, default=100)
    group.add_argument("--eval_iter", help="No. of steps per evaluation", type=int, default=100)
    group.add_argument("--save_iter", help="No. of steps per saving", type=int, default=100)
    # debug params
    group.add_argument("--do_profile", help="Whether to do profiling", action="store_true")
    group.add_argument("--record_shapes", help="Whether to record tensor shapes for profiling", action="store_true")

    return parser


def add_predict_args(parser):
    group = parser.add_argument_group("Prediction options")
    group.add_argument("--do_predict", help="Whether to do prediction", action="store_true")
    group.add_argument("--do_score", help="Whether to score predictions", action="store_true")
    group.add_argument("--checkpoint_step_start", help="First checkpoint step", type=int)
    group.add_argument("--checkpoint_step_end", help="Last checkpoint step", type=int)
    group.add_argument("--predict_batch_size", help="Batch size for prediction", type=int, default=4096)
    # decoding params
    group.add_argument("--test_bin", help="Test npz", type=str, default="")
    group.add_argument("--result_file", help="Result file", type=str, default="")
    group.add_argument("--beam_size", help="Beam size for decoding", type=int, default=0)
    group.add_argument("--n_best", help="Number of best results to be retained", type=int, default=10)
    group.add_argument("--temperature", help="Beam search temperature", type=float, default=1.0)
    group.add_argument("--predict_min_len", help="Min length for prediction", type=int, default=1)
    group.add_argument("--predict_max_len", help="Max length for prediction", type=int, default=512)
