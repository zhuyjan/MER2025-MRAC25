import matplotlib.pyplot as plt
import os
import os.path as osp


def draw_loss(epoch, train_loss, valid_loss, save_path):
    plt.figure()
    plt.plot(epoch, train_loss, label="train loss")
    plt.plot(epoch, valid_loss, label="valid loss")
    plt.legend()
    plt.savefig(
        save_path,
        bbox_inches="tight",
        dpi=500,
    )
    plt.close()
    return


def draw_metric(epoch, metric, key, save_path):
    plt.figure()
    plt.plot(epoch, metric, label="valid {}".format(key))
    plt.legend()
    plt.savefig(
        save_path,
        bbox_inches="tight",
        dpi=500,
    )
    plt.close()
