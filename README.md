# More Is Better: A MoE-Based Emotion Recognition Framework with Human Preference Alignment (MRAC @ ACM-MM 2025)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2508.06036-B21A1B)](https://arxiv.org/abs/2508.06036)
[![GitHub Stars](https://img.shields.io/github/stars/zhuyjan/MER2025-MRAC25?style=social)](https://github.com/zhuyjan/MER2025-MRAC25/stargazers)

This repository provides our <strong><font color="red">runner-up</font></strong> solution for [MER2025-SEMI challenge](https://zeroqiaoba.github.io/MER2025-website/) at MRAC'25 workshop. If our project helps you, please give us a star ⭐ on GitHub to support us. 🙏🙏

## Overview
<p align="center"><img src="images/fig1.png" alt="method" width="1000px" /></p>

We propose a comprehensive framework, grounded in the principle that "more is better," to construct a robust Mixture of Experts (MoE) emotion recognition system. Our approach integrates a diverse range of input modalities as independent experts, including novel signals such as knowledge from large Vision-Language Models (VLMs) and temporal Action Unit (AU) information.

## Setup
Our environment setup is identical to that of [MERBench](https://github.com/zeroQiaoba/MERTools/tree/master/MERBench).
```bash
conda env create -f environment.yml
```

## Workflow

**Preprocessing**
1. Extract features for each sample. Each sample should correspond to a `.npy` file with shape `(C,)`.  
   You can refer to [MER2025 Track1](https://github.com/zeroQiaoba/MERTools/tree/master/MER2025/MER2025_Track1) for more details on feature extraction. 
   (We also provide our Gemini prompt for reference — see `prompt.txt` for details.)
2. Run `./preprocess/check_feat_dim.py` to obtain the dimensionality of each type of feature.
3. Fill in the feature names and their corresponding dimensions in the modality-specific dictionaries in `./config.py`.
4. Run `./preprocess/mer2025_base.py` to split the dataset into training and validation sets.

**Training**
- Run `./train.py` to perform training and validation.
- Run `./run_eval_unimodal.sh` to evaluate unimodal performance. (All three branches use the same input for fair comparison with the trimodal setting.)

## Training Information Format

Saved as `.npy` files under `./train_lst/`, each containing a Python dictionary with the following structure:

```python
train_info = {
    # List of video names, e.g., ["example1", ...]
    "names": split_train_names, 
    # List of emotion labels, e.g., ["happy", ...]
    "emos": split_train_emos,                
    # List of emotion valence values, e.g., [-10, ...]
    "vals": split_train_vals,
}

valid_info = {
    # List of video names, e.g., ["example1", ...]
    "names": split_valid_names, 
    # List of emotion labels, e.g., ["happy", ...]
    "emos": split_valid_emos,                
    # List of emotion valence values, e.g., [-10, ...]
    "vals": split_valid_vals,
}
```

## Citation
If you find this work useful for your research, please give us a star and use the following BibTeX entry for citation.
```
@inproceedings{xie2025more,
  title={More is better: A moe-based emotion recognition framework with human preference alignment},
  author={Xie, Jun and Zhu, Yingjian and Chen, Feng and Zhang, Zhenghao and Fan, Xiaohui and Yi, Hongzhu and Wang, Xinming and Yu, Chen and Bi, Yue and Zhao, Zhaoran and others},
  booktitle={Proceedings of the 3rd International Workshop on Multimodal and Responsible Affective Computing},
  pages={2--7},
  year={2025}
}
```
