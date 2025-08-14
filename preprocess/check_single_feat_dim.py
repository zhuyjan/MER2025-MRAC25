import os
import os.path as osp
from glob import glob

import numpy as np

if __name__ == "__main__":
    feat_root = "/mnt/public/share/data/MER2025/mer2025-dataset-process/features/InternVL_2_5_HiCo_R16-UTT"
    npy_files = glob(osp.join(feat_root, "*.npy"))

    print(f"Found {len(npy_files)} npy files in {feat_root}")
    print("-" * 50)
    
    for npy_file in npy_files:
        file_name = osp.basename(npy_file)
        try:
            data = np.load(npy_file)
            print(f"{file_name}: shape={data.shape}")
        except Exception as e:
            print(f"{file_name}: Error loading file - {e}")
