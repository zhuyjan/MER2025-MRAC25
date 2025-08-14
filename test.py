import argparse
import datetime
import logging
import os
import os.path as osp
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

import config
from dataloader.dataloader import get_dataloader_train, get_dataloader_valid
from models import get_models
from toolkit.utils.draw_process import draw_loss, draw_metric
from toolkit.utils.eval import calculate_results
from toolkit.utils.functions import func_update_storage, merge_args_config
from toolkit.utils.loss import CELoss, MSELoss
from toolkit.utils.metric import gain_metric_from_results

# emotion rule
# emotions = ["neutral", "angry", "happy", "sad", "worried", "surprise"]


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def test_model(
    args,
    model,
    dataloader,
):

    vidnames = []
    val_preds, val_labels = [], []
    emo_probs, emo_labels = [], []
    losses = []

    model.eval()

    pbar = tqdm(range(len(dataloader)), desc=f"test")

    for iter, data in enumerate(dataloader):

        # read data + cuda
        audios, texts, videos, emos, vals, bnames = data

        vidnames += bnames

        batch = {}
        batch["videos"] = videos.float().cuda()
        batch["audios"] = audios.float().cuda()
        batch["texts"] = texts.float().cuda()

        emos = emos.long().cuda()
        vals = vals.float().cuda()

        if args.train_input_mode == "input_gt":
            _, emos_out, _, _ = model([batch, emos])
        elif args.train_input_mode == "input":
            _, emos_out, _, _ = model(batch)

        emo_probs.append(emos_out.data.cpu().numpy())
        emo_labels.append(emos.data.cpu().numpy())

        pbar.update(1)

    pbar.close()

    if emo_probs != []:
        emo_probs = np.concatenate(emo_probs)
    if emo_labels != []:
        emo_labels = np.concatenate(emo_labels)
    if val_preds != []:
        val_preds = np.concatenate(val_preds)
    if val_labels != []:
        val_labels = np.concatenate(val_labels)
    results, _ = calculate_results(emo_probs, emo_labels, val_preds, val_labels)
    save_results = dict(
        names=vidnames,
        **results,
    )

    y_true = []
    y_pred = []
    emo_preds = np.argmax(emo_probs, 1)
    for emo_pred, emo_label in zip(emo_preds, emo_labels):
        y_pred.append(emotions[emo_pred])
        y_true.append(emotions[emo_label])
    save_results["emotions"] = emotions
    conf_matrix = confusion_matrix(y_true, y_pred, labels=emotions)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    print("emotions: " + str(emotions))
    print("Confusion Matrix:")
    print(str(conf_matrix))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return save_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )

    # --- data config
    parser.add_argument(
        "--dataset",
        type=str,
        default="MER24-test_3A_whisper-base-UTT",
        help="dataset info name",
    )
    parser.add_argument(
        "--emo_rule",
        type=str,
        default="MER",
        help="emo map function from emotion to index",
    )

    # --- save config
    parser.add_argument(
        "--save_root",
        type=str,
        default="/mnt/public/gxj/EmoNets/saved_test",
        help="save prediction results and models",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="test",
        help="save prediction results and models",
    )

    # --- model config
    parser.add_argument(
        "--model",
        type=str,
        default="attention",
        help="model name for training [attention, mer_rank5, and others]",
    )
    parser.add_argument(
        "--load_key",
        type=str,
        default="/mnt/public/gxj/EmoNets/saved/seed0_MER24_3A_whisper-base-UTT/attention/2025-04-06-21-26-53",
        help="keyword about which model weight to load",
    )

    # --- test sets
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        metavar="BS",
        help="batch size",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, metavar="nw", help="number of workers"
    )
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")

    args = parser.parse_args()

    set_seed(args.seed)

    torch.cuda.set_device(args.gpu)

    # 若没有关键词，则使用load key的路径
    if args.load_key not in config.MODEL_DIR_DICT[args.model].keys():
        config.MODEL_DIR_DICT[args.model][args.load_key] = args.load_key

    # 用当前设置参数覆盖训练时的同名设置
    load_args_path = osp.join(
        config.MODEL_DIR_DICT[args.model][args.load_key], "best_args.npy"
    )
    load_args = np.load(load_args_path, allow_pickle=True).item()["args"]
    load_args_dic = vars(load_args)
    args_dic = vars(args)
    for key in args_dic:
        load_args_dic[key] = args_dic[key]
    args = argparse.Namespace(**load_args_dic)  # 两部分参数重叠

    print("====== set save dir =======")
    now = datetime.datetime.now()
    save_dir = os.path.join(args.save_root, args.dataset, args.model, args.save_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("{}\n".format(save_dir))

    logging_path = osp.join(save_dir, "info.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        filename=logging_path,
        filemode="w",
    )

    logging.info("====== load info and config =======")
    dataset_info = np.load(
        osp.join(config.PATH_TO_TEST_LST, args.dataset + ".npy"), allow_pickle=True
    ).item()

    for feat_name in ["video", "audio", "text"]:
        logging.info(
            "Input feature: {} ===> dim is (1, {})".format(
                dataset_info["feat_dim"][f"{feat_name}_feat_description"],
                dataset_info["feat_dim"][f"{feat_name}_dim"],
            )
        )

    emo_rule = config.EMO_RULE[args.emo_rule]
    emotions = list(emo_rule)

    logging.info("====== load dataset =======")
    test_info = dataset_info["test_info"]
    test_loader = get_dataloader_valid(
        names=test_info["names"],
        emo_labels=test_info["emos"],
        val_labels=test_info["vals"],
        emo_rule=emo_rule,
        audio_feat_paths=test_info["audio_feat"],
        video_feat_paths=test_info["video_feat"],
        text_feat_paths=test_info["text_feat"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    logging.info("====== build model =======")
    model = get_models(args).cuda()
    assert args.load_key is not None
    model_path = osp.join(config.MODEL_DIR_DICT[args.model][args.load_key], "best.pth")
    model.load_state_dict(
        torch.load(model_path, map_location=f"cuda:0", weights_only=True)
    )

    logging.info("load model: {} \n".format(args.model))
    logging.info("load model weight: {} \n".format(model_path))

    logging.info("====== Evaluation =======")
    best_eval_metric = None  # for select best model weight
    record_test = {"emoacc": [], "emofscore": []}

    test_results = test_model(
        args,
        model,
        test_loader,
    )

    # np.save(osp.join(save_dir, "results.npy"), test_results)
