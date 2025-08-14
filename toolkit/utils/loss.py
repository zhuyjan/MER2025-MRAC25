import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


# classification loss
class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction="sum")

    def forward(self, pred, target):
        pred = F.log_softmax(pred, 1)  # [n_samples, n_classes]
        target = target.long()  # [n_samples]
        loss = self.loss(pred, target) / len(pred)
        return loss


# regression loss
class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction="sum")

    def forward(self, pred, target):
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)
        loss = self.loss(pred, target) / len(pred)
        return loss


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lambda_c=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes  # 类别数
        self.feat_dim = feat_dim  # 特征维度
        self.lambda_c = lambda_c  # 平衡系数
        # 初始化每个类别的特征中心
        # self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centers = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, x, labels):
        """
        x: 当前批次样本的特征向量 (batch_size, feat_dim)
        labels: 当前批次样本的类别标签 (batch_size)
        """
        batch_size = x.size(0)

        # 取出当前批次样本对应类别的中心
        centers_batch = self.centers.cuda().index_select(0, labels)

        # 计算 Center Loss
        center_loss = (
            self.lambda_c * 0.5 * torch.sum((x - centers_batch) ** 2) / batch_size
        )
        return center_loss


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.2, 0.3, 0.5], gamma=2, reduction="mean", classnum=None):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        if classnum is None:
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = torch.tensor([1.0 / classnum] * classnum)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha.cuda()[
            target
        ]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(
            pred, dim=1
        )  # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(
            log_softmax, dim=1, index=target.view(-1, 1)
        )  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(
            logpt
        )  # 对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = (
            alpha * (1 - pt) ** self.gamma * ce_loss
        )  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, device, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        print("LDAM weights:", weight)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.device = device
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)
        # criterion = LDAMLoss(cls_num_list=a list of numer of samples in each class, max_m=0.5, s=30, weight=per_cls_weights)
        """
        max_m: represents the margin used in the loss function. It controls the separation between different classes in the feature space.
        The appropriate value for max_m depends on the specific dataset and the severity of the class imbalance. 
        You can start with a small value and gradually increase it to observe the impact on the model's performance. 
        If the model struggles with class separation or experiences underfitting, increasing max_m might help. However,
        be cautious not to set it too high, as it can cause overfitting or make the model too conservative.
       
        s:  higher s value results in sharper probabilities (more confident predictions), while a lower s value 
        leads to more balanced probabilities. In practice, a larger s value can help stabilize training and improve convergence,
        especially when dealing with difficult optimization landscapes.
        The choice of s depends on the desired scale of the logits and the specific requirements of your problem. 
        It can be used to adjust the balance between the margin and the original logits. A larger s value amplifies 
        the impact of the logits and can be useful when dealing with highly imbalanced datasets. 
        You can experiment with different values of s to find the one that works best for your dataset and model.

        """


class LMFLoss(nn.Module):
    def __init__(
        self,
        cls_num_list,
        device,
        weight=None,
        alpha=0.2,
        beta=0.2,
        gamma=2,
        max_m=0.8,
        s=5,
        add_LDAM_weigth=False,
    ):
        super().__init__()
        self.focal_loss = MultiClassFocalLossWithAlpha(classnum=len(cls_num_list))
        if add_LDAM_weigth:
            LDAM_weight = weight
        else:
            LDAM_weight = None
        print(
            "LMF loss: alpha: ",
            alpha,
            " beta: ",
            beta,
            " gamma: ",
            gamma,
            " max_m: ",
            max_m,
            " s: ",
            s,
            " LDAM_weight: ",
            add_LDAM_weigth,
        )
        self.ldam_loss = LDAMLoss(cls_num_list, device, max_m, weight=LDAM_weight, s=s)
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        focal_loss_output = self.focal_loss(output, target)
        ldam_loss_output = self.ldam_loss(output, target)
        total_loss = self.alpha * focal_loss_output + self.beta * ldam_loss_output
        return total_loss
