import os
import os.path as osp
from glob import glob

import numpy as np

if __name__ == "__main__":
    feat_root = "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/"
    flst = glob(osp.join(feat_root, "*"))

    for feat_dir in flst:
        feat_name = osp.basename(feat_dir)
        fpath = glob(osp.join(feat_dir, "*"))[0]
        feat_dim = np.load(fpath).shape[0]

        print("{}, dim={}".format(feat_name, feat_dim))
