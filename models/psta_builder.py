"""Pyramid Spatial-Temporal Aggregation (PSTA) for Video-based Re-Identification.

Adapted from Wang et al., "Pyramid Spatial-Temporal Aggregation for Video-based Person Re-Identification", ICCV 2021.
Preserves all original architecture, parameters, and loss formulations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


MODEL_URLS = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def init_pretrained_weight(model, model_url):
    """Initializes model with pretrained weights."""
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


# --- ResNet-50 Backbone with last_stride=1 ---

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# --- Temporal Reference Attention (TRA) ---

class TRA(nn.Module):

    def __init__(self, inplanes=1024, num='1'):
        super(TRA, self).__init__()
        self.inplanes = inplanes
        self.num = num
        self.relu = nn.ReLU(True)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.gamma_temporal = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=int(inplanes / 8), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(inplanes / 8)),
            self.relu
        )
        self.gamma_temporal.apply(weights_init_kaiming)

        self.beta_temporal = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=int(inplanes / 8), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(inplanes / 8)),
            self.relu
        )
        self.beta_temporal.apply(weights_init_kaiming)

        # Original spatial dimension: 2 * 16 * 8 = 256 for 256x128 pedestrian resolution
        self.gg_temporal = nn.Sequential(
            nn.Conv2d(in_channels=2 * 16 * 8, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
        )
        self.gg_temporal.apply(weights_init_kaiming)

        self.tte_para = nn.Sequential(
            nn.Conv2d(in_channels=2 * 128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
        )
        self.tte_para.apply(weights_init_kaiming)

        self.te_para = nn.Sequential(
            nn.Conv2d(in_channels=2 * 128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.te_para.apply(weights_init_kaiming)

        self.theta_channel = nn.Sequential(
            nn.Conv1d(in_channels=inplanes, out_channels=int(inplanes / 8), kernel_size=1, stride=1, padding=0, bias=False),
            self.relu,
        )
        self.theta_channel.apply(weights_init_kaiming)

        self.channel_para = nn.Sequential(
            nn.Linear(in_features=int(inplanes / 4), out_features=int(inplanes / 8)),
            self.relu,
            nn.Linear(in_features=int(inplanes / 8), out_features=inplanes),
            nn.Sigmoid()
        )
        self.channel_para.apply(weights_init_kaiming)

    def forward(self, featmap, re_featmap, vect_featmap, embed_feat):
        b, t, c, h, w = featmap.size()
        gamma_feat = self.gamma_temporal(re_featmap).view(b, t, -1, h * w)
        beta_feat = self.beta_temporal(re_featmap).view(b, t, -1, h * w)
        channel_para = self.theta_channel(vect_featmap.permute(0, 2, 1))
        gap_feat_map0 = []

        for idx in range(0, t, 2):
            para0 = torch.cat((channel_para[:, :, idx], channel_para[:, :, idx + 1]), 1)
            para_00 = self.channel_para(para0).view(b, -1, 1, 1)
            para1 = torch.cat((channel_para[:, :, idx + 1], channel_para[:, :, idx]), 1)
            para_01 = self.channel_para(para1).view(b, -1, 1, 1)

            embed_feat0 = embed_feat[:, idx, :, :, :]
            embed_feat1 = embed_feat[:, idx + 1, :, :, :]

            gamma_feat0 = gamma_feat[:, idx, :, :].permute(0, 2, 1)
            beta_feat0 = beta_feat[:, idx + 1, :, :]
            Gs0 = torch.matmul(gamma_feat0, beta_feat0)
            Gs_in0 = Gs0.permute(0, 2, 1).view(b, h * w, h, w)
            Gs_out0 = Gs0.view(b, h * w, h, w)

            gamma_feat1 = gamma_feat[:, idx + 1, :, :].permute(0, 2, 1)
            beta_feat1 = beta_feat[:, idx, :, :]
            Gs1 = torch.matmul(gamma_feat1, beta_feat1)
            Gs_in1 = Gs1.permute(0, 2, 1).view(b, h * w, h, w)
            Gs_out1 = Gs1.view(b, h * w, h, w)

            Gs_joint0 = torch.cat((Gs_in0, Gs_out1), 1)
            Gs_joint0 = self.gg_temporal(Gs_joint0)
            para_alpha = self.tte_para(torch.cat((embed_feat0, embed_feat1), 1))
            para_alpha = self.te_para(torch.cat((para_alpha, Gs_joint0), 1))

            Gs_joint1 = torch.cat((Gs_in1, Gs_out0), 1)
            Gs_joint1 = self.gg_temporal(Gs_joint1)
            para_beta = self.tte_para(torch.cat((embed_feat1, embed_feat0), 1))
            para_beta = self.te_para(torch.cat((para_beta, Gs_joint1), 1))

            para_00 = para_00 * para_alpha
            para_01 = para_01 * para_beta

            gap_map0 = para_00 * featmap[:, idx, :, :, :] + para_01 * featmap[:, idx + 1, :, :, :]
            gap_map0 = self.relu(gap_map0)
            gap_map0 = gap_map0 ** 2
            gap_feat_map0.append(gap_map0)

        gap_feat_map0 = torch.stack(gap_feat_map0, 1)
        return gap_feat_map0


# --- Spatial Reference Attention (SRA) ---

class SRA(nn.Module):

    def __init__(self, inplanes=1024, num='1'):
        super(SRA, self).__init__()
        self.inplanes = inplanes
        self.num = num
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.alphi_appearance = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=int(inplanes / 8), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(inplanes / 8)),
            self.relu
        )
        self.alphi_appearance.apply(weights_init_kaiming)

        self.delta_appearance = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=int(inplanes / 8), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(inplanes / 8)),
            self.relu
        )
        self.delta_appearance.apply(weights_init_kaiming)

        # Original spatial dimension: 2 * 16 * 8 = 256 for 256x128 pedestrian resolution
        self.gg_spatial = nn.Sequential(
            nn.Conv2d(in_channels=2 * 16 * 8, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
        )
        self.gg_spatial.apply(weights_init_kaiming)

        self.spa_para = nn.Sequential(
            nn.Conv2d(in_channels=2 * 128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            self.sigmoid
        )
        self.spa_para.apply(weights_init_kaiming)

        self.app_channel = nn.Sequential(
            nn.Linear(in_features=inplanes, out_features=int(inplanes / 8)),
            self.relu,
            nn.Linear(in_features=int(inplanes / 8), out_features=inplanes),
            self.sigmoid
        )
        self.app_channel.apply(weights_init_kaiming)

    def forward(self, feat_map, re_featmap, Embeding_feature, feat_vect, aggregative_feature=None):
        b, t, c, h, w = feat_map.size()
        Embeding_feat = Embeding_feature.view(b * t, -1, h, w)
        alphi_feat = self.alphi_appearance(re_featmap).view(b * t, -1, h * w)
        delta_feat = self.delta_appearance(re_featmap).view(b * t, -1, h * w)
        alphi_feat = alphi_feat.permute(0, 2, 1)
        Gs = torch.matmul(alphi_feat, delta_feat)
        Gs_in = Gs.permute(0, 2, 1).view(b * t, h * w, h, w)
        Gs_out = Gs.view(b * t, h * w, h, w)
        Gs_joint = torch.cat((Gs_in, Gs_out), 1)
        Gs_joint = self.gg_spatial(Gs_joint)
        para_spa = torch.cat((Embeding_feat, Gs_joint), 1)
        para_spa = self.spa_para(para_spa).view(b, t, -1, h, w)

        aggregative_feature_list = []
        for i in range(0, t, 2):
            para_0 = self.app_channel(feat_vect[:, i, :]).view(b, -1, 1, 1)
            para_1 = self.app_channel(feat_vect[:, i + 1, :]).view(b, -1, 1, 1)

            para_0 = para_0 * para_spa[:, i, :, :, :]
            para_1 = para_1 * para_spa[:, i + 1, :, :, :]

            aggregative_feature_list.append(aggregative_feature[:, int(i / 2), :, :, :] + self.relu(para_0 * feat_map[:, i, :, :, :] + para_1 * feat_map[:, i + 1, :, :, :]))

        aggregative_features = torch.stack(aggregative_feature_list, 1)
        aggregative_features = aggregative_features.view(b * aggregative_features.size(1), -1, h, w)
        return aggregative_features


# --- Spatial-Temporal Aggregation Module (STAM) ---

class STAM(nn.Module):

    def __init__(self, inplanes=1024, mid_planes=256, num='1', **kwargs):
        super(STAM, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.num = num

        self.Embeding = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=mid_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            self.relu
        )
        self.Embeding.apply(weights_init_kaiming)

        self.TRAG = TRA(inplanes=inplanes, num=num)
        self.SRAG = SRA(inplanes=inplanes, num=num)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=mid_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=mid_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=inplanes, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes),
            self.relu
        )
        self.conv_block.apply(weights_init_kaiming)

    def forward(self, feat_map):
        b, t, c, h, w = feat_map.size()
        reshape_map = feat_map.view(b * t, c, h, w)
        feat_vect = self.avg(reshape_map).view(b, t, -1)
        embed_feat = self.Embeding(reshape_map).view(b, t, -1, h, w)

        gap_feat_map0 = self.TRAG(feat_map, reshape_map, feat_vect, embed_feat)
        gap_feat_map = self.SRAG(feat_map, reshape_map, embed_feat, feat_vect, gap_feat_map0)
        gap_feat_map = self.conv_block(gap_feat_map)
        gap_feat_map = gap_feat_map.view(b, -1, c, h, w)
        return gap_feat_map


# --- PSTA Losses (Cosine Triplet + Label-smoothed Cross-Entropy) ---

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets_onehot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smoothed = (1 - self.epsilon) * targets_onehot + self.epsilon / self.num_classes
        loss = (-targets_smoothed * log_probs).mean(0).sum()
        return loss


class CosineTripletLoss(nn.Module):

    def __init__(self, margin=0.3):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        fnorm = torch.norm(inputs, p=2, dim=1, keepdim=True)
        l2norm = inputs.div(fnorm.clamp(min=1e-12))
        dist = -torch.mm(l2norm, l2norm.t())

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            # Positives for anchor i (excluding self if other genuine positives exist)
            pos_mask = mask[i].clone()
            if pos_mask.sum() > 1:
                pos_mask[i] = False
            dist_ap.append(dist[i][pos_mask].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


# --- Complete VideoPSTA Model Wrapper ---

class VideoPSTA(nn.Module):

    def __init__(self, cfg):
        super(VideoPSTA, self).__init__()

        self.num_classes = getattr(cfg, "num_classes", 0)
        self.seq_len = getattr(cfg, "clip_len", 8)
        self.in_planes = 2048
        self.plances = 1024
        self.mid_channel = 256

        self.base = ResNet(last_stride=1)
        init_pretrained_weight(self.base, MODEL_URLS['resnet50'])
        print('PSTA: Loaded pretrained ImageNet ResNet-50 weights.')

        self.avg_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.down_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_planes, out_channels=self.plances, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.plances),
            self.relu
        )

        t = self.seq_len
        self.layer1 = STAM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, num='1')
        t = t / 2
        self.layer2 = STAM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, num='2')
        t = t / 2
        self.layer3 = STAM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, num='3')

        self.bottleneck = nn.ModuleList([nn.BatchNorm1d(self.plances) for _ in range(3)])
        self.classifier = nn.ModuleList([nn.Linear(self.plances, self.num_classes) for _ in range(3)])

        for bn in self.bottleneck:
            bn.bias.requires_grad_(False)
            bn.apply(weights_init_kaiming)

        for cl in self.classifier:
            cl.apply(weight_init_classifier)

        # Losses matching PSTA's default setup
        margin = getattr(cfg, "margin", 0.3)
        self.xent = CrossEntropyLabelSmooth(num_classes=self.num_classes, epsilon=0.1)
        self.tent = CosineTripletLoss(margin=margin)

    def forward(self, x, targets=None, **kwargs):
        """
        x: (B, T, C, H, W)
        Returns:
            train: (BN_feature_list, cls_score)
            eval: BN_feature_list[2] (1024-D)
        """
        b, t, c, h, w = x.size()
        x_reshaped = x.view(b * t, c, h, w)
        feat_map = self.base(x_reshaped)  # (b * t, 2048, 16, 8)
        feat_h = feat_map.size(2)
        feat_w = feat_map.size(3)

        feat_map = self.down_channel(feat_map)
        feat_map = feat_map.view(b, t, -1, feat_h, feat_w)
        feature_list = []
        pyr_list = []

        feat_map_1 = self.layer1(feat_map)
        feature_1 = torch.mean(feat_map_1, 1)
        feature1 = self.avg_2d(feature_1).view(b, -1)
        feature_list.append(feature1)
        pyr_list.append(feature1)

        feat_map_2 = self.layer2(feat_map_1)
        feature_2 = torch.mean(feat_map_2, 1)
        feature_2 = self.avg_2d(feature_2).view(b, -1)
        pyr_list.append(feature_2)

        feature2 = torch.stack(pyr_list, 1)
        feature2 = torch.mean(feature2, 1)
        feature_list.append(feature2)

        feat_map_3 = self.layer3(feat_map_2)
        feature_3 = torch.mean(feat_map_3, 1)
        feature_3 = self.avg_2d(feature_3).view(b, -1)
        pyr_list.append(feature_3)

        feature3 = torch.stack(pyr_list, 1)
        feature3 = torch.mean(feature3, 1)
        feature_list.append(feature3)

        BN_feature_list = []
        for i in range(len(feature_list)):
            BN_feature_list.append(self.bottleneck[i](feature_list[i]))

        if not self.training:
            # During evaluation, return the deepest pyramid Level-3 representation (1024-D)
            return BN_feature_list[2]

        cls_score = []
        for i in range(len(BN_feature_list)):
            cls_score.append(self.classifier[i](BN_feature_list[i]))

        return BN_feature_list, cls_score

    def compute_loss(self, outputs, targets):
        """
        Deep supervision loss over the 3 pyramid levels:
        Average of Cross-Entropy and Cosine Triplet Loss across all 3 heads.
        Inputs are cast to float32 for numerical stability under mixed precision.
        """
        BN_feature_list, cls_score = outputs

        BN_feature_list = [f.float() for f in BN_feature_list]
        cls_score = [s.float() for s in cls_score]

        loss_ce = sum(self.xent(score, targets) for score in cls_score) / len(cls_score)
        loss_tri = sum(self.tent(feat, targets) for feat in BN_feature_list) / len(BN_feature_list)
        total_loss = loss_ce + loss_tri

        return total_loss, loss_tri, loss_ce
