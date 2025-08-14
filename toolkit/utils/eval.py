import random
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, accuracy_score


def calculate_results(emo_probs=[], emo_labels=[], val_preds=[], val_labels=[]):

    emo_preds = np.argmax(emo_probs, 1)
    emo_accuracy = accuracy_score(emo_labels, emo_preds)
    emo_fscore = f1_score(emo_labels, emo_preds, average="weighted")

    results = {
        "emoprobs": emo_probs,
        "emolabels": emo_labels,
        "emoacc": emo_accuracy,
        "emofscore": emo_fscore,
    }
    outputs = f"f1:{emo_fscore:.4f}_acc:{emo_accuracy:.4f}"
    return results, outputs
