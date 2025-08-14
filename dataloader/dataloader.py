"""
Description: build [train / valid / test] dataloader
Author: Xiongjun Guan
Date: 2023-03-01 19:41:05
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2023-03-20 21:01:47

Copyright (C) 2023 by Xiongjun Guan, Tsinghua University. All rights reserved.
"""

import copy
import logging
import os.path as osp
import random

import cv2
import numpy as np
from scipy import signal
from torch.utils.data import DataLoader, Dataset


class load_dataset_train(Dataset):

    def __init__(
        self,
        names,
        emo_labels,
        val_labels,
        emo_rule,
        feat_roots,
    ):
        self.names = names
        self.emo_labels = emo_labels
        self.val_labels = val_labels
        self.feat_roots = feat_roots

        self.emo2idx_mer, self.idx2emo_mer = {}, {}
        for ii, emo in enumerate(emo_rule):
            self.emo2idx_mer[emo] = ii
        for ii, emo in enumerate(emo_rule):
            self.idx2emo_mer[ii] = emo

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        emo = self.emo2idx_mer[self.emo_labels[idx]]
        val = self.val_labels[idx]
        name = self.names[idx]

        inputs = {}
        idx = 0
        for feat_root in self.feat_roots:
            ftitle = osp.basename(name).split(".")[0]
            inputs["feat{}".format(idx)] = (
                np.load(osp.join(feat_root, ftitle + ".npy"))
                .squeeze()
                .astype(np.float32)
            )
            idx += 1

        return inputs, emo, val, name


def get_dataloader_train(
    names,
    emo_labels,
    val_labels,
    emo_rule,
    feat_roots,
    batch_size=1,
    num_workers=4,
    shuffle=True,
):
    # Create dataset
    try:
        dataset = load_dataset_train(
            names=names,
            emo_labels=emo_labels,
            val_labels=val_labels,
            emo_rule=emo_rule,
            feat_roots=feat_roots,
        )
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    logging.info(f"n_train:{len(dataset)}")

    return train_loader


def get_dataloader_valid(
    names,
    emo_labels,
    val_labels,
    emo_rule,
    feat_roots,
    batch_size=1,
    num_workers=4,
    shuffle=False,
):
    # Create dataset
    try:
        dataset = load_dataset_train(
            names=names,
            emo_labels=emo_labels,
            val_labels=val_labels,
            emo_rule=emo_rule,
            feat_roots=feat_roots,
        )
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    logging.info(f"n_valid:{len(dataset)}")

    return valid_loader
