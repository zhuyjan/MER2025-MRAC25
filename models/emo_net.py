import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(
        self, inp_dim, out_dim, dropout, mid_dim=None, num_heads=4, num_layers=2
    ):

        super(Encoder, self).__init__()
        if isinstance(inp_dim, list):
            self.mode = len(inp_dim)
        else:
            self.mode = 1

        if mid_dim is None:
            mid_dim = out_dim
        if self.mode == 1:
            # self.norm = nn.BatchNorm1d(in_size)
            self.linear = nn.Linear(inp_dim, mid_dim)
            self.net = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=mid_dim,
                    nhead=num_heads,
                    dim_feedforward=2048,
                    dropout=dropout,
                    batch_first=True,
                ),
                num_layers=num_layers,
            )
            self.linear2 = nn.Linear(mid_dim, out_dim)
        elif self.mode == 2:
            self.linear = nn.Linear(35, mid_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, mid_dim))
            self.net = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=mid_dim,
                    nhead=num_heads,
                    dim_feedforward=2048,
                    dropout=dropout,
                    batch_first=True,
                ),
                num_layers=num_layers,
            )
            self.linear2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        """
        Args:
            x: tensor of shape (batch_size, in_size)
        """
        # normed = self.norm(x)

        if self.mode == 1:
            x = self.linear(x)
            x = self.net(x)
        elif self.mode == 2:
            x = x[:, :, 674:]  # only AU feature in openface features
            x = self.linear(x)
            B, L, C = x.shape
            cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, mid_dim)
            x = torch.cat((cls_token, x), dim=1)  # (B, L+1, mid_dim)
            x = self.net(x)  # (B, L+1, mid_dim)
            x = x[:, 0, :]

        x = self.linear2(x)

        return x


class FcClassifier(nn.Module):
    def __init__(self, hidden_dim, cls_layers, output_dim=6, dropout=0, use_bn=True):
        super(FcClassifier, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(hidden_dim, cls_layers[0]))
        if use_bn is True:
            self.fc_layers.append(nn.BatchNorm1d(cls_layers[0]))
        self.fc_layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            self.fc_layers.append(nn.Dropout(dropout))

        for i in range(1, len(cls_layers)):
            self.fc_layers.append(nn.Linear(cls_layers[i - 1], cls_layers[i]))
            if use_bn is True:
                self.fc_layers.append(nn.BatchNorm1d(cls_layers[i]))
            self.fc_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                self.fc_layers.append(nn.Dropout(dropout))

        self.output_layer = nn.Linear(cls_layers[-1], output_dim)

    def forward(self, x):
        for layer in self.fc_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class Auto_WeightV1(nn.Module):
    def __init__(self, args):
        super(Auto_WeightV1, self).__init__()
        feat_dims = args.feat_dims
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout = args.dropout
        hidden_dim = args.hidden_dim
        self.grad_clip = args.grad_clip
        self.feat_dims = feat_dims

        num_heads = args.num_heads
        num_layers = args.num_layers

        assert args.feat_type == "utt"

        assert len(feat_dims) <= 3 * 6
        if len(feat_dims) >= 1:
            self.encoder0 = Encoder(
                feat_dims[0],
                hidden_dim,
                dropout,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.cls0 = FcClassifier(
                hidden_dim,
                [256, 128],
                output_dim=output_dim1,
                dropout=0,
                use_bn=True,
            )
        if len(feat_dims) >= 2:
            self.encoder1 = Encoder(
                feat_dims[1],
                hidden_dim,
                dropout,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.cls1 = FcClassifier(
                hidden_dim,
                [256, 128],
                output_dim=output_dim1,
                dropout=0,
                use_bn=True,
            )
        if len(feat_dims) >= 3:
            self.encoder2 = Encoder(
                feat_dims[2],
                hidden_dim,
                dropout,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.cls2 = FcClassifier(
                hidden_dim,
                [256, 128],
                output_dim=output_dim1,
                dropout=0,
                use_bn=True,
            )
        if len(feat_dims) >= 4:
            self.encoder3 = Encoder(
                feat_dims[3],
                hidden_dim,
                dropout,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.cls3 = FcClassifier(
                hidden_dim,
                [256, 128],
                output_dim=output_dim1,
                dropout=0,
                use_bn=True,
            )
        if len(feat_dims) >= 5:
            self.encoder4 = Encoder(
                feat_dims[4],
                hidden_dim,
                dropout,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.cls4 = FcClassifier(
                hidden_dim,
                [256, 128],
                output_dim=output_dim1,
                dropout=0,
                use_bn=True,
            )
        if len(feat_dims) >= 6:
            self.encoder5 = Encoder(
                feat_dims[5],
                hidden_dim,
                dropout,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.cls5 = FcClassifier(
                hidden_dim,
                [256, 128],
                output_dim=output_dim1,
                dropout=0,
                use_bn=True,
            )
        if len(feat_dims) >= 7:
            self.encoder6 = Encoder(
                feat_dims[6],
                hidden_dim,
                dropout,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.cls6 = FcClassifier(
                hidden_dim,
                [256, 128],
                output_dim=output_dim1,
                dropout=0,
                use_bn=True,
            )
        if len(feat_dims) >= 8:
            self.encoder7 = Encoder(
                feat_dims[7],
                hidden_dim,
                dropout,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.cls7 = FcClassifier(
                hidden_dim,
                [256, 128],
                output_dim=output_dim1,
                dropout=0,
                use_bn=True,
            )
        if len(feat_dims) >= 9:
            self.encoder8 = Encoder(
                feat_dims[8],
                hidden_dim,
                dropout,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.cls8 = FcClassifier(
                hidden_dim,
                [256, 128],
                output_dim=output_dim1,
                dropout=0,
                use_bn=True,
            )
        if len(feat_dims) >= 10:
            self.encoder9 = Encoder(
                feat_dims[9],
                hidden_dim,
                dropout,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.cls9 = FcClassifier(
                hidden_dim,
                [256, 128],
                output_dim=output_dim1,
                dropout=0,
                use_bn=True,
            )
        if len(feat_dims) >= 11:
            self.encoder10 = Encoder(
                feat_dims[10],
                hidden_dim,
                dropout,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.cls10 = FcClassifier(
                hidden_dim,
                [256, 128],
                output_dim=output_dim1,
                dropout=0,
                use_bn=True,
            )
        if len(feat_dims) >= 12:
            self.encoder11 = Encoder(
                feat_dims[11],
                hidden_dim,
                dropout,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.cls11 = FcClassifier(
                hidden_dim,
                [256, 128],
                output_dim=output_dim1,
                dropout=0,
                use_bn=True,
            )
        if len(feat_dims) >= 13:
            self.encoder12 = Encoder(
                feat_dims[12],
                hidden_dim,
                dropout,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.cls12 = FcClassifier(
                hidden_dim,
                [256, 128],
                output_dim=output_dim1,
                dropout=0,
                use_bn=True,
            )
        if len(feat_dims) >= 14:
            self.encoder13 = Encoder(
                feat_dims[13],
                hidden_dim,
                dropout,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.cls13 = FcClassifier(
                hidden_dim,
                [256, 128],
                output_dim=output_dim1,
                dropout=0,
                use_bn=True,
            )
        if len(feat_dims) >= 15:
            self.encoder14 = Encoder(
                feat_dims[14],
                hidden_dim,
                dropout,
                num_heads=num_heads,
                num_layers=num_layers,
            )
            self.cls14 = FcClassifier(
                hidden_dim,
                [256, 128],
                output_dim=output_dim1,
                dropout=0,
                use_bn=True,
            )

        # --- feature attention
        self.attention_mlp = Encoder(
            hidden_dim * len(feat_dims),
            hidden_dim,
            dropout,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.fc_att = nn.Linear(hidden_dim, len(feat_dims))

        # --- logit weight
        self.weight_fc = nn.Sequential(
            nn.Linear(hidden_dim * len(feat_dims), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(feat_dims) + 1),
        )

        self.fc_out_1 = FcClassifier(
            hidden_dim * len(feat_dims),
            [256, 128],
            output_dim=output_dim1,
            dropout=0,
            use_bn=True,
        )
        self.fc_out_2 = FcClassifier(
            hidden_dim * len(feat_dims),
            [256, 128],
            output_dim=output_dim2,
            dropout=0,
            use_bn=True,
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inp):
        """
        support feat_type: utt | frm-align | frm-unalign
        """

        [batch, emos] = inp

        hiddens = []
        logits = []
        if len(self.feat_dims) >= 1:
            hiddens.append(self.encoder0(batch[f"feat0"]))
            logits.append(self.cls0(hiddens[0]))
        if len(self.feat_dims) >= 2:
            hiddens.append(self.encoder1(batch[f"feat1"]))
            logits.append(self.cls1(hiddens[1]))
        if len(self.feat_dims) >= 3:
            hiddens.append(self.encoder2(batch[f"feat2"]))
            logits.append(self.cls2(hiddens[2]))
        if len(self.feat_dims) >= 4:
            hiddens.append(self.encoder3(batch[f"feat3"]))
            logits.append(self.cls3(hiddens[3]))
        if len(self.feat_dims) >= 5:
            hiddens.append(self.encoder4(batch[f"feat4"]))
            logits.append(self.cls4(hiddens[4]))
        if len(self.feat_dims) >= 6:
            hiddens.append(self.encoder5(batch[f"feat5"]))
            logits.append(self.cls5(hiddens[5]))
        if len(self.feat_dims) >= 7:
            hiddens.append(self.encoder6(batch[f"feat6"]))
            logits.append(self.cls6(hiddens[6]))
        if len(self.feat_dims) >= 8:
            hiddens.append(self.encoder7(batch[f"feat7"]))
            logits.append(self.cls7(hiddens[7]))
        if len(self.feat_dims) >= 9:
            hiddens.append(self.encoder8(batch[f"feat8"]))
            logits.append(self.cls8(hiddens[8]))
        if len(self.feat_dims) >= 10:
            hiddens.append(self.encoder9(batch[f"feat9"]))
            logits.append(self.cls9(hiddens[9]))
        if len(self.feat_dims) >= 11:
            hiddens.append(self.encoder10(batch[f"feat10"]))
            logits.append(self.cls10(hiddens[10]))
        if len(self.feat_dims) >= 12:
            hiddens.append(self.encoder11(batch[f"feat11"]))
            logits.append(self.cls11(hiddens[11]))
        if len(self.feat_dims) >= 13:
            hiddens.append(self.encoder12(batch[f"feat12"]))
            logits.append(self.cls12(hiddens[12]))
        if len(self.feat_dims) >= 14:
            hiddens.append(self.encoder13(batch[f"feat13"]))
            logits.append(self.cls13(hiddens[13]))
        if len(self.feat_dims) >= 15:
            hiddens.append(self.encoder14(batch[f"feat14"]))
            logits.append(self.cls14(hiddens[14]))

        multi_hidden1 = torch.cat(hiddens, dim=1)  # [32, 384]

        logits.append(self.fc_out_1(multi_hidden1))
        vals_out = self.fc_out_2(multi_hidden1)

        weight = self.weight_fc(multi_hidden1).unsqueeze(-1)

        multi_logits = torch.stack(logits, dim=2).permute(0, 2, 1)  # [32, N, 6]
        emos_out = (weight * multi_logits).sum(dim=1)

        if self.training:
            interloss = self.cal_loss(
                logits,
                emos,
            )
        else:
            interloss = torch.tensor(0).cuda()

        return None, emos_out, vals_out, interloss

    def cal_loss(
        self,
        logits,
        emos,
    ):
        # feat_A_concat, feat_V_concat, feat_L_concat):
        emos = emos.to(logits[0].device)

        loss = 0
        for logit in logits:
            loss += self.criterion(logit, emos) / len(logits)

        return loss
