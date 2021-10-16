import argparse
import gdown
import os


urls_fns_dict = {
    "USPTO_50k": [
        ("https://drive.google.com/uc?id=1pz-qkfeXzeD_drO9XqZVGmZDSn20CEwr", "src-train.txt"),
        ("https://drive.google.com/uc?id=1ZmmCJ-9a0nHeQam300NG5i9GJ3k5lnUl", "tgt-train.txt"),
        ("https://drive.google.com/uc?id=1NqLI3xpy30kH5fbVC0l8bMsMxLKgO-5n", "src-val.txt"),
        ("https://drive.google.com/uc?id=19My9evSNc6dlk9od5OrwkWauBpzL_Qgy", "tgt-val.txt"),
        ("https://drive.google.com/uc?id=1l7jSqYfIr0sL5Ad6TUxsythqVFjFudIx", "src-test.txt"),
        ("https://drive.google.com/uc?id=17ozyajoqPFeVjfViI59-QpVid1M0zyKN", "tgt-test.txt")
    ],
    "USPTO_full": [
        ("https://drive.google.com/uc?id=1PbHoIYbm7-69yPOvRA0CrcjojGxVCJCj", "src-train.txt"),
        ("https://drive.google.com/uc?id=1RRveZmyXAxufTEix-WRjnfdSq81V9Ud9", "tgt-train.txt"),
        ("https://drive.google.com/uc?id=1jOIA-20zFhQ-x9fco1H7Q10R6CfxYeZo", "src-val.txt"),
        ("https://drive.google.com/uc?id=19ZNyw7hLJaoyEPot5ntKBxz_o-_R14QP", "tgt-val.txt"),
        ("https://drive.google.com/uc?id=1ErtNB29cpSld8o_gr84mKYs51eRat0H9", "src-test.txt"),
        ("https://drive.google.com/uc?id=1kV9p1_KJm8EqK6OejSOcqRsO8DwOgjL_", "tgt-test.txt")
    ],
    "USPTO_480k": [
        ("https://drive.google.com/uc?id=1RysNBvB2rsMP0Ap9XXi02XiiZkEXCrA8", "src-train.txt"),
        ("https://drive.google.com/uc?id=1CxxcVqtmOmHE2nhmqPFA6bilavzpcIlb", "tgt-train.txt"),
        ("https://drive.google.com/uc?id=1FFN1nz2yB4VwrpWaBuiBDzFzdX3ONBsy", "src-val.txt"),
        ("https://drive.google.com/uc?id=1pYCjWkYvgp1ZQ78EKQBArOvt_2P1KnmI", "tgt-val.txt"),
        ("https://drive.google.com/uc?id=10t6pHj9yR8Tp3kDvG0KMHl7Bt_TUbQ8W", "src-test.txt"),
        ("https://drive.google.com/uc?id=1FeGuiGuz0chVBRgePMu0pGJA4FVReA-b", "tgt-test.txt")
    ],
    "USPTO_STEREO": [
        ("https://drive.google.com/uc?id=1r3_7WMEor7-CgN34Foj-ET-uFco0fURU", "src-train.txt"),
        ("https://drive.google.com/uc?id=1HUBLDtqEQc6MQ-FZQqNhh2YBtdc63xdG", "tgt-train.txt"),
        ("https://drive.google.com/uc?id=1WwCH8ASgBM1yOmZe0cJ46bj6kPSYYIRc", "src-val.txt"),
        ("https://drive.google.com/uc?id=19OsSpXxWJ-XWuDwfG04VTYzcKAJ28MTw", "tgt-val.txt"),
        ("https://drive.google.com/uc?id=1FcbWZnyixhptaO6DIVjCjm_CeTomiCQJ", "src-test.txt"),
        ("https://drive.google.com/uc?id=1rVWvbmoVC90jyGml_t-r3NhaoWVVSKLe", "tgt-test.txt")
    ]
}


def parse_args():
    parser = argparse.ArgumentParser("download_raw_data.py", conflict_handler="resolve")
    parser.add_argument("--data_name", help="data name", type=str, default="",
                        choices=["USPTO_50k", "USPTO_full", "USPTO_480k", "USPTO_STEREO"])

    return parser.parse_args()


def main():
    args = parse_args()
    data_path = os.path.join("./data", args.data_name)

    os.makedirs(data_path, exist_ok=True)

    for url, fn in urls_fns_dict[args.data_name]:
        ofn = os.path.join(data_path, fn)
        if not os.path.exists(ofn):
            gdown.download(url, ofn, quiet=False)
            assert os.path.exists(ofn)
        else:
            print(f"{ofn} exists, skip downloading")


if __name__ == "__main__":
    main()
