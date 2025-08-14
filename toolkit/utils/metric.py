import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, accuracy_score


# 综合维度和离散的评价指标
def overall_metric(emo_fscore, val_mse):
    final_score = emo_fscore - val_mse * 0.25
    return final_score


# 只返回 metric 值，用于模型筛选
def gain_metric_from_results(eval_results, metric_name="emoval"):

    if metric_name == "emoval":
        fscore = eval_results["emofscore"]
        valmse = eval_results["valmse"]
        overall = overall_metric(fscore, valmse)
        sort_metric = overall
    elif metric_name == "emo":
        fscore = eval_results["emofscore"]
        sort_metric = fscore
    elif metric_name == "val":
        valmse = eval_results["valmse"]
        sort_metric = -valmse
    elif metric_name == "loss":
        loss = eval_results["loss"]
        sort_metric = -loss

    return sort_metric
