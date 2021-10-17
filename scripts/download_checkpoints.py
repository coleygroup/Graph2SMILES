import argparse
import gdown
import os


urls_fns_dict = {
    "USPTO_50k_dgat.pt": "https://drive.google.com/uc?id=1HNxzI4Sn4TqjI70LtDVrdy1k1qPzX9UJ",
    "USPTO_50k_dgcn.pt": "https://drive.google.com/uc?id=11IJzTvyEvLclVSDN6gz2wy1Pg2hSOqa0",
    "USPTO_full_dgat.pt": "https://drive.google.com/uc?id=165_S9GWNssrP8VeAHYIHqgISqueigTua",
    "USPTO_full_dgcn.pt": "https://drive.google.com/uc?id=1L9HEmg-bbp6oPbWdeKCwSRjkHGWMk8DD",
    "USPTO_480k_dgat.pt": "https://drive.google.com/uc?id=1F37CGO2WpNUYHhPph0GiRA2eHIRiUK5o",
    "USPTO_480k_dgcn.pt": "https://drive.google.com/uc?id=1eFtTd5OTvVa9MMpUI4B3Y3izDjMV_k8z",
    "USPTO_STEREO_dgat.pt": "https://drive.google.com/uc?id=1lky5Sq6CYn-Mhezf-SW2yfs0yPYNx2Pm",
    "USPTO_STEREO_dgcn.pt": "https://drive.google.com/uc?id=1S8v-UAWlWJEL7KQkzPijj4RoDmB51kFe"
}


def parse_args():
    parser = argparse.ArgumentParser("download_raw_data.py", conflict_handler="resolve")
    parser.add_argument("--data_name", help="data name", type=str, default="",
                        choices=["USPTO_50k", "USPTO_full", "USPTO_480k", "USPTO_STEREO"])
    parser.add_argument("--mpn_type", help="type of mpn used", type=str, default="",
                        choices=["dgat", "dgcn"])

    return parser.parse_args()


def main():
    args = parse_args()

    ofn = f"{args.data_name}_{args.mpn_type}.pt"
    url = urls_fns_dict[ofn]

    ofn = os.path.join("./checkpoints/pretrained", ofn)
    if not os.path.exists(ofn):
        gdown.download(url, ofn, quiet=False)
        assert os.path.exists(ofn)
    else:
        print(f"{ofn} exists, skip downloading")


if __name__ == "__main__":
    main()
