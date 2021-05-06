import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import precision_recall_fscore_support


def get_weighted_bce_loss(prediction, gt):
    loss = nn.BCELoss(reduction='none')

    class_loss = loss(prediction, gt)

    weights = torch.ones_like(gt)
    w_negative = gt.sum() / gt.size(0)
    w_positive = 1 - w_negative

    weights[gt >= 0.5] = w_positive
    weights[gt < 0.5] = w_negative
    w_class_loss = torch.mean(weights * class_loss)

    #######################################
    # get classification precision and recall
    predicted_labels = prediction.detach().cpu().round().numpy()
    cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels, average='binary')

    return w_class_loss, cls_precision, cls_recall


def get_circle_loss(coords_dist, feats_dist, pos_radius, safe_radius, pos_optimal=0.1, neg_optimal=1.4, log_scale=16):
    """
    Modified from: https://github.com/XuyangBai/D3Feat.pytorch
    """
    pos_margin, neg_margin = pos_optimal, neg_optimal
    pos_mask = coords_dist < pos_radius
    neg_mask = coords_dist > safe_radius

    ## get anchors that have both positive and negative pairs
    row_sel = ((pos_mask.sum(-1) > 0) * (neg_mask.sum(-1) > 0)).detach()
    col_sel = ((pos_mask.sum(-2) > 0) * (neg_mask.sum(-2) > 0)).detach()

    # get alpha for both positive and negative pairs
    pos_weight = feats_dist - 1e5 * (~pos_mask).float()  # mask the non-positive
    pos_weight = (pos_weight - pos_optimal)  # mask the uninformative positive
    pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach()

    neg_weight = feats_dist + 1e5 * (~neg_mask).float()  # mask the non-negative
    neg_weight = (neg_optimal - neg_weight)  # mask the uninformative negative
    neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight).detach()

    lse_pos_row = torch.logsumexp(log_scale * (feats_dist - pos_margin) * pos_weight, dim=-1)
    lse_pos_col = torch.logsumexp(log_scale * (feats_dist - pos_margin) * pos_weight, dim=-2)

    lse_neg_row = torch.logsumexp(log_scale * (neg_margin - feats_dist) * neg_weight, dim=-1)
    lse_neg_col = torch.logsumexp(log_scale * (neg_margin - feats_dist) * neg_weight, dim=-2)

    loss_row = F.softplus(lse_pos_row + lse_neg_row) / log_scale
    loss_col = F.softplus(lse_pos_col + lse_neg_col) / log_scale

    circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

    return circle_loss


def get_contrastive_loss(fxd, fxm, match):
    mp = torch.Tensor([.1]).cuda()
    mn = torch.Tensor([1.4]).cuda()
    fxd_r = torch.stack([fxd] * fxm.shape[0])
    fxm_r = torch.stack([fxm] * fxd.shape[0]).transpose(0, 1)
    mask = match*1e9
    fdists_all = torch.norm(fxd_r - fxm_r, dim=2).t()
    fdists_all = fdists_all+mask
    '''
    fdists_all =
    ||fxd[0]-fxm[0] ||fxd[0]-fxm[1]|| ... ||fxd[0]-fxm[N]||
    ||fxd[1]-fxm[0] ||fxd[1]-fxm[1]|| ... ||fxd[1]-fxm[N]|| 
    .                                       .
    .                                       .
    .                                       .
    ||fxd[N]-fxm[0] ||fxd[N]-fxm[1]|| ... ||fxd[N]-fxm[N]||
    '''

    fdm_mins, fdm_argmins = torch.min(fdists_all, dim=1)
    fmd_mins, fmd_argmins = torch.min(fdists_all, dim=0)

    # fdists_pos = torch.norm(fxd - fxm, dim=1)
    fdists_pos = torch.norm(fxd_r - fxm_r, dim=2).t().contiguous().view(-1)[match.view(-1) == 1]

    a = F.relu(fdists_pos - mp).pow(2).sum() / len(fdists_pos)
    b = torch.mean(F.relu(mn - fdm_mins).pow(2))
    c = torch.mean(F.relu(mn - fmd_mins).pow(2))

    l = a + (b + c) / 2

    return l, fdists_pos, torch.median(fdists_all, dim=1), torch.median(fdists_all, dim=0)