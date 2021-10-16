import gdown
import os

urls_fns = [
    # for generating negative examples
    ("https://drive.google.com/uc?id=1zIN0T0tG-F1QDwb6QM7EJpUMl1Bw899f",
     "50k_clean_rxnsmi_noreagent_train.pickle"),
    ("https://drive.google.com/uc?id=1fxKp92MzsTha4Gd2wt721ZSakylKy70o",
     "50k_clean_rxnsmi_noreagent_valid.pickle"),
    ("https://drive.google.com/uc?id=1jcVgfSP_kG7DrNjyIJNOcH_MPkjWtAJR",
     "50k_clean_rxnsmi_noreagent_test.pickle"),
    ("https://drive.google.com/uc?id=1tyXIa_f20jzA8J5pvudSdwZ-t3dRnP8V",
     "50k_mol_smis.pickle"),
    # ("https://drive.google.com/uc?id=1hSaXJB97YypV7Pav1qB0ovh1_oQLZs3t",
    #  "50k_mol_smi_to_sparse_fp_idx.pickle"),
    # ("https://drive.google.com/uc?id=1XesKizw5E5IBXTTcIVNenz1H8QD69uyn",
    #  "50k_sparse_fp_idx_to_mol_smi.pickle"),
    # ("https://drive.google.com/uc?id=12ZQPPYdugx7WDKjuXnSyHg6f6ILsn5sx",
    #  "50k_count_mol_fps.npz"),
    # ("https://drive.google.com/uc?id=1BLvjp5LjlPJg8W9KvJ3pcXiEE5alWL0M",
    #  "50k_cosine_count.bin"),
    # ("https://drive.google.com/uc?id=1iGrqy99TNBrHzRmLbSdchjgQ0yBnS2S7",
    #  "50k_cosine_count.bin.dat"),
    ("https://drive.google.com/uc?id=1rZoCn70np-5dfRown0wtnM54Iq-XZ3xk",
     "50k_neg150_rad2_maxsize3_mutprodsmis.pickle"),
    # pre-computed augmented data, for FeedforwardEBM
    # ("https://drive.google.com/uc?id=1kAuwfGv0s1OOo9be0NyNNhOekdWCwGLT",
    #  "50k_rdm_5_cos_5_bit_5_1_1_mut_10_train.npz"),
    # ("https://drive.google.com/uc?id=1BhcIeVsSSmRXpfCfTqsorUXWg_Tw5i7a",
    #  "50k_rdm_5_cos_5_bit_5_1_1_mut_10_valid.npz"),
    # ("https://drive.google.com/uc?id=13DwNxixNp_ylOTuA047mZSTgTCKL9WYm",
    #  "50k_rdm_5_cos_5_bit_5_1_1_mut_10_test.npz"),
    # retrosim CSV files
    ("https://drive.google.com/uc?id=15Le31UeYosYXC7-9PXG2tu7jAP2lY2cY",
     "retrosim_200maxtest_200maxprec_train.csv"),
    ("https://drive.google.com/uc?id=1-QkcmuLXxAI-lRbUdZ5c0mYDevz0x47d",
     "retrosim_200maxtest_200maxprec_valid.csv"),
    ("https://drive.google.com/uc?id=1DQi_dXs2l8rrPae6zQdAWsKXvkSreTlg",
     "retrosim_200maxtest_200maxprec_test.csv"),
    # retrosim rxn fp files, for FeedforwardEBM
    # ("https://drive.google.com/uc?id=1mBp6eBYr9InGf1bPdyM9wyJyuCUt_fdB",
    #  "retrosim_rxn_fps_train.npz"),
    # ("https://drive.google.com/uc?id=1KmKGfOSYs4HU5hpPGvvP5BaAj85iQKD4",
    #  "retrosim_rxn_fps_valid.npz"),
    # ("https://drive.google.com/uc?id=1JzbyLC74N8r3O6ULGC26t_hmFfLZ3FTe",
    #  "retrosim_rxn_fps_test.npz")
    # # GLN CSV files
    # ("https://drive.google.com/uc?id=1MTyyZZH0lEy-P83IyNSRcvXGqCzdlnVA",
    # "GLN_200topk_200maxk_200beam_test.csv"),
    # ("https://drive.google.com/uc?id=1snuxs43NVavVtnyt600rHyX5hlc7X_IC",
    # "GLN_200topk_200maxk_200beam_train.csv"),
    # ("https://drive.google.com/uc?id=17zDvlXpBYAoOf8Qo3ujIPhg93NZu83-I",
    # "GLN_200topk_200maxk_200beam_valid.csv"),

]
output = "./rxnebm/data/cleaned_data/"

for url, fn in urls_fns:
    ofn = os.path.join(output, fn)
    if not os.path.exists(ofn):
        gdown.download(url, output, quiet=False)
        assert os.path.exists(ofn)
    else:
        print(f"{ofn} exists, skip downloading")
