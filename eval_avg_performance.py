import os
import os.path as osp

import numpy as np

if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    feats = [
        "InternVL_2_5_HiCo_R16-UTT",
    ]
    # feats = [
    #     "chinese-hubert-large-UTT",
    #     "Qwen2-Audio-7B-UTT",
    #     "chinese-hubert-base-UTT",
    #     "whisper-large-v2-UTT",
    #     "chinese-wav2vec2-large-UTT",
    #     "chinese-wav2vec2-base-UTT",
    #     "wavlm-base-UTT",
    # ]
    # feats = [
    #     "chinese-roberta-wwm-ext-large-UTT",
    #     "chinese-roberta-wwm-ext-UTT",
    #     "chinese-macbert-large-UTT",
    #     "chinese-macbert-base-UTT",
    #     "bloom-7b1-UTT",
    # ]

    for feat in feats:
        avg_acc = 0
        avg_fscore = 0
        cnt = 0
        for seed in seeds:
            fdir = osp.join(
                "./saved", feat + "_3", "seed" + str(seed), "auto_attention", "run"
            )

            res = np.load(
                osp.join(fdir, "best_valid_results.npy"), allow_pickle=True
            ).item()

            avg_acc += res["eval_emoacc"]
            avg_fscore += res["eval_emofscore"]
            cnt += 1

        avg_acc /= cnt
        avg_fscore /= cnt
        print("### {}".format(feat + "_3"))
        print("acc={:.4f}, fscore={:.4f}".format(avg_acc, avg_fscore))
        print("")

        a = 1
