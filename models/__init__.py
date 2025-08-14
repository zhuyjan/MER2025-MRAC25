import torch.nn as nn

from .auto_attention import Auto_Attention


class get_models(nn.Module):
    def __init__(self, args):
        super(get_models, self).__init__()
        # misa/mmim在有些参数配置下会存在梯度爆炸的风险
        # tfn 显存占比比较高

        MODEL_MAP = {
            # 特征压缩到句子级再处理，所以支持 utt/align/unalign
            "auto_attention": Auto_Attention,
        }
        self.model = MODEL_MAP[args.model](args)

    def forward(self, batch):
        return self.model(batch)
