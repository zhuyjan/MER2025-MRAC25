import argparse
import ast
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
from tqdm import tqdm

import config
from dataloader.dataloader import get_dataloader_train, get_dataloader_valid
from models import get_models
from toolkit.utils.draw_process import draw_loss, draw_metric
from toolkit.utils.eval import calculate_results
from toolkit.utils.functions import func_update_storage, merge_args_config
from toolkit.utils.loss import CELoss, MSELoss
from toolkit.utils.metric import gain_metric_from_results


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_model(model, save_path):
    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        model_state,
        osp.join(save_path),
    )
    return


def train_or_eval_model(
    args, model, reg_loss, cls_loss, dataloader, epoch, optimizer=None, train=False
):

    if not train:
        vidnames = []
        val_preds, val_labels = [], []
        emo_probs, emo_labels = [], []
    losses = []

    assert not train or optimizer != None

    if train:
        model.train()
    else:
        model.eval()

    if train:
        pbar = tqdm(range(len(dataloader)), desc=f"epoch:{epoch+1}, train")
    else:
        pbar = tqdm(range(len(dataloader)), desc=f"epoch:{epoch+1}, valid")

    for iter, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()

        # read data + cuda
        inputs, emos, vals, bnames = data

        if not train:
            vidnames += bnames

        for k in inputs.keys():
            inputs[k] = inputs[k].cuda()

        emos = emos.long().cuda()
        vals = vals.float().cuda()

        # forward process
        # start_time = time.time()

        if args.train_input_mode == "input_gt":
            features, emos_out, vals_out, interloss = model([inputs, emos])
        elif args.train_input_mode == "input":
            features, emos_out, vals_out, interloss = model(inputs)
        # duration = time.time() - start_time
        # macs, params = profile(model, inputs=(batch, ))
        # print(f"MACs: {macs}, Parameters: {params}, Duration: {duration}; bsize: {len(bnames)}")

        # loss calculation
        loss = interloss

        if args.output_dim1 != 0:
            loss = loss + cls_loss(emos_out, emos)
            if not train:
                emo_probs.append(emos_out.data.cpu().numpy())
                emo_labels.append(emos.data.cpu().numpy())
        if args.output_dim2 != 0:
            loss = loss + reg_loss(vals_out, vals)
            if not train:
                val_preds.append(vals_out.data.cpu().numpy())
                val_labels.append(vals.data.cpu().numpy())
        losses.append(loss.data.cpu().numpy())

        pbar.set_postfix(**{"loss": loss.item()})
        pbar.update(1)

        # optimize params
        if train:
            loss.backward()
            if model.model.grad_clip != -1:
                torch.nn.utils.clip_grad_value_(
                    [param for param in model.parameters() if param.requires_grad],
                    model.model.grad_clip,
                )
            optimizer.step()

    pbar.close()

    if not train:
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
            loss=np.mean(losses),
            **results,
        )
    else:
        save_results = dict(
            loss=np.mean(losses),
        )
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
        default="seed1",
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
        default="./saved",
        help="save prediction results and models",
    )
    parser.add_argument(
        "--save_as_time",
        action="store_true",
        default=False,
        help="save suffix as time, default: run",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=False,
        help="whether to save model, default: False",
    )

    # --- model config
    parser.add_argument(
        "--model",
        type=str,
        default="auto_attention",
        help="model name for training ",
    )
    parser.add_argument(
        "--model_pretrain",
        type=str,
        default=None,
        help="pretrained model path",
    )

    # --- feat config
    parser.add_argument(
        "--feat",
        type=str,
        default='["senet50face_UTT","senet50face_UTT","senet50face_UTT"]',
        help="use feat",
    )

    # --- train config
    parser.add_argument(
        "--lr", type=float, default=1e-4, metavar="lr", help="set lr rate"
    )
    # parser.add_argument(
    #     "--lr_end", type=float, default=1e-5, metavar="lr", help="set lr rate"
    # )
    parser.add_argument(
        "--l2",
        type=float,
        default=0.00001,
        metavar="L2",
        help="L2 regularization weight",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        metavar="BS",
        help="batch size [deal with OOM]",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, metavar="nw", help="number of workers"
    )
    parser.add_argument(
        "--epochs", type=int, default=60, metavar="E", help="number of epochs"
    )
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")

    args = parser.parse_args()
    args.feat = ast.literal_eval(args.feat)

    set_seed(args.seed)

    torch.cuda.set_device(args.gpu)

    print("====== set save dir =======")
    now = datetime.datetime.now()
    if args.save_as_time:
        save_dir = os.path.join(
            args.save_root, args.dataset, args.model, now.strftime("%Y-%m-%d-%H-%M-%S")
        )
    else:
        save_dir = os.path.join(args.save_root, args.dataset, args.model, "run")
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
        osp.join(config.PATH_TO_TRAIN_LST, args.dataset + ".npy"), allow_pickle=True
    ).item()

    model_config = OmegaConf.load("models/model-tune.yaml")[args.model]
    model_config = OmegaConf.to_container(model_config, resolve=True)

    emo_rule = config.EMO_RULE[args.emo_rule]
    model_config["emo_rule"] = emo_rule
    model_config["output_dim1"] = len(emo_rule)
    model_config["output_dim2"] = 0

    feat_dims = []
    feat_roots = []
    feat_types = []
    for use_feat in args.feat:
        if use_feat in config.FEAT_VIDEO_DICT:
            feat_dims.append(config.FEAT_VIDEO_DICT[use_feat][0])
            feat_roots.append(config.FEAT_VIDEO_DICT[use_feat][1])
            feat_types.append("V")
        elif use_feat in config.FEAT_AUDIO_DICT:
            feat_dims.append(config.FEAT_AUDIO_DICT[use_feat][0])
            feat_roots.append(config.FEAT_AUDIO_DICT[use_feat][1])
            feat_types.append("A")
        elif use_feat in config.FEAT_TEXT_DICT:
            feat_dims.append(config.FEAT_TEXT_DICT[use_feat][0])
            feat_roots.append(config.FEAT_TEXT_DICT[use_feat][1])
            feat_types.append("T")
    model_config["feat_dims"] = feat_dims
    model_config["feat_roots"] = feat_roots
    for feat_type, feat_dim, feat_root in zip(feat_types, feat_dims, feat_roots):
        logging.info("Modality:{}, Input feature: {}".format(feat_type, feat_root))
        logging.info("===> dim is (1, {}) \n".format(feat_dim))

    args = merge_args_config(args, model_config)  # 两部分参数重叠
    logging.info("save config: {} \n".format(osp.join(save_dir, "args.yaml")))
    OmegaConf.save(vars(args), osp.join(save_dir, "args.yaml"))

    logging.info("====== load dataset =======")
    train_info = dataset_info["train_info"]
    train_loader = get_dataloader_train(
        names=train_info["names"],
        emo_labels=train_info["emos"],
        val_labels=train_info["vals"],
        emo_rule=emo_rule,
        feat_roots=args.feat_roots,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    valid_info = dataset_info["valid_info"]
    valid_loader = get_dataloader_valid(
        names=valid_info["names"],
        emo_labels=valid_info["emos"],
        val_labels=valid_info["vals"],
        emo_rule=emo_rule,
        feat_roots=args.feat_roots,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    logging.info("====== build model =======")
    model = get_models(args).cuda()
    cls_loss = CELoss().cuda()
    reg_loss = MSELoss().cuda()
    optimizer = optim.Adam(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.l2,
    )
    logging.info("load model: {} \n".format(args.model))

    if args.model_pretrain is not None:
        model.load_state_dict(torch.load(args.model_pretrain, map_location=f"cuda:0"))

    logging.info("====== Training and Evaluation =======")
    best_eval_metric = None  # for select best model weight
    record_train = {"epoch": [], "loss": []}
    record_valid = {"epoch": [], "loss": [], "emoacc": [], "emofscore": []}
    best_valid = {}
    for epoch in range(args.epochs):
        logging.info(
            "epoch: {}, lr:{:.8f}".format(
                epoch + 1, optimizer.state_dict()["param_groups"][0]["lr"]
            )
        )

        train_results = train_or_eval_model(
            args,
            model,
            reg_loss,
            cls_loss,
            train_loader,
            epoch=epoch,
            optimizer=optimizer,
            train=True,
        )
        valid_results = train_or_eval_model(
            args,
            model,
            reg_loss,
            cls_loss,
            valid_loader,
            epoch=epoch,
            optimizer=None,
            train=False,
        )

        # --- logging info
        logging_info = "\tTRAIN: loss:{:.4f}".format(train_results["loss"])
        logging.info(logging_info)

        logging_info = "\tVALID: loss:{:.4f}, acc:{:.4f}, f-score:{:.4f}".format(
            valid_results["loss"],
            valid_results["emoacc"],
            valid_results["emofscore"],
        )
        logging.info(logging_info)

        # --- record data and plot
        record_train["epoch"].append(epoch + 1)
        record_train["loss"].append(train_results["loss"])

        record_valid["epoch"].append(epoch + 1)
        for key in record_valid:
            if key == "epoch":
                continue
            record_valid[key].append(valid_results[key])

        # --- save best model
        ## select metric
        eval_metric = valid_results["emofscore"]

        # save eval results, model config and weight
        if best_eval_metric is None or best_eval_metric < eval_metric:
            best_eval_metric = eval_metric
            # save best result info in valid dataset
            best_valid["epoch"] = epoch + 1
            for key in record_valid:
                if key in ["epoch"]:
                    continue
                best_valid[key] = record_valid[key][-1]

            # save best result in valid dataset
            epoch_store = {}
            func_update_storage(
                inputs=valid_results, prefix="eval", outputs=epoch_store
            )
            np.save(osp.join(save_dir, "best_valid_results.npy"), epoch_store)

            # save best model weight and args
            np.save(osp.join(save_dir, "best_args.npy"), {"args": args})
            if args.save_model:
                save_path = f"{save_dir}/best.pth"
                save_model(model, save_path)

            logging.info("\t*** Update best info ! ***")

    draw_loss(
        record_train["epoch"],
        record_train["loss"],
        record_valid["loss"],
        osp.join(save_dir, "loss.png"),
    )
    for key in record_valid:
        if key in ["epoch", "loss"]:
            continue
        draw_metric(
            record_valid["epoch"],
            record_valid[key],
            key,
            osp.join(save_dir, "{}.png".format(key)),
        )

    logging.info("End Training ! \n\n")
    logging_info = "BEST: epoch:{:.0f}, loss:{:.4f}, acc:{:.4f}, f-score:{:.4f}".format(
        best_valid["epoch"],
        best_valid["loss"],
        best_valid["emoacc"],
        best_valid["emofscore"],
    )
    logging.info(logging_info)

    # clear memory
    del model
    del optimizer
    torch.cuda.empty_cache()
