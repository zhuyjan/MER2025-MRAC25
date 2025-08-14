import glob
import os
import shutil
import sys

import pandas as pd

sys.path.append("/mnt/public/gxj_2/EmoNet_Pro/")
import argparse
import os.path as osp
import random
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

import config


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# run -d toolkit/preprocess/mer2024.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )
    args = parser.parse_args()

    random_seed = args.seed
    set_seed(random_seed)
    print("### random seed: {}".format(random_seed))

    # --- data config
    emo_rule = "MER"
    gt_path = "/mnt/public/share/data/MER2025/mer2025-dataset/track1_train_disdim.csv"

    # --- save config
    save_root = "/mnt/public/gxj_2/EmoNet_Pro/lst_train/mer25_train_val"
    save_name = "seed{}".format(random_seed)

    # --- load data and format
    mapping_rules = config.EMO_RULE[emo_rule]

    all_names = []
    all_emos = []
    all_vals = []
    df = pd.read_csv(gt_path)
    for index, row in tqdm(df.iterrows(), total=len(df)):
        name = row["name"]
        discrete = row["discrete"]
        valence = row["valence"]

        # emo = mapping_rules.index(discrete)

        all_names.append(name)
        all_emos.append(discrete)
        all_vals.append(valence)

    counted_numbers = {}
    for emo in all_emos:
        if emo in counted_numbers:
            counted_numbers[emo] += 1
        else:
            counted_numbers[emo] = 1
    for emo in mapping_rules:
        print(
            "{}, num={}, percent={:.2f}%".format(
                emo,
                counted_numbers[emo],
                100 * counted_numbers[emo] / len(all_emos),
            )
        )

    # --------------- split train & test ---------------
    split_ratio = 0.2
    whole_num = len(all_names)

    # gain indices for cross-validation
    indices = np.arange(whole_num)
    random.shuffle(indices)

    # split indices into 1-fold
    each_folder_num = int(whole_num * split_ratio)
    valid_idxs = indices[0:each_folder_num]
    train_idxs = indices[each_folder_num:]

    split_train_names = []
    split_train_emos = []
    split_train_vals = []
    for idx in train_idxs:
        split_train_names.append(all_names[idx])
        split_train_emos.append(all_emos[idx])
        split_train_vals.append(all_vals[idx])

    split_valid_names = []
    split_valid_emos = []
    split_valid_vals = []
    for idx in valid_idxs:
        split_valid_names.append(all_names[idx])
        split_valid_emos.append(all_emos[idx])
        split_valid_vals.append(all_vals[idx])

    train_info = {
        "names": split_train_names,
        "emos": split_train_emos,
        "vals": split_train_vals,
    }
    valid_info = {
        "names": split_valid_names,
        "emos": split_valid_emos,
        "vals": split_valid_vals,
    }

    print("----------- summary")
    print(
        "split_train: {}\nsplit_valid: {}".format(
            len(train_info["names"]), len(valid_info["names"])
        )
    )

    save_path = os.path.join(save_root, save_name + ".npy")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    np.save(
        save_path,
        {
            "train_info": train_info,
            "valid_info": valid_info,
        },
    )
