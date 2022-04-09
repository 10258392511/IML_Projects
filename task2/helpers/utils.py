import numpy as np
import torch

from scipy.special import expit
from sklearn.metrics import roc_auc_score, mean_squared_error, make_scorer


def auc_star_loss(y_pred, y, epoch_pred, epoch_gt, gamma=0.3, p=2):
    # y_pred, y: (B,); epoch_pred, epoch_gt: (N,)
    y = (y > 0.5)  # convert to torch.bool
    epoch_gt = (epoch_gt > 0.5)
    y_pred_pos = y_pred[y]
    y_pred_neg = y_pred[torch.logical_not(y)]
    epoch_pred_pos = epoch_pred[epoch_gt]
    epoch_pred_neg = epoch_pred[torch.logical_not(epoch_gt)]
    inds = np.arange(len(epoch_pred_pos))
    np.random.shuffle(inds)
    epoch_pred_pos_samples = epoch_pred_pos[inds[:y_pred_neg.shape[0]]]  # (B_neg,)
    inds = np.arange(len(epoch_pred_neg))
    np.random.shuffle(inds)
    epoch_pred_neg_samples = epoch_pred_neg[inds[:y_pred_pos.shape[0]]]  # (B_pos,)

    loss_pos, loss_neg = 0, 0
    if y_pred_pos.shape[0] > 0:
        dist = epoch_pred_neg_samples - y_pred_pos.unsqueeze(1)  # (B_pos, B_pos)
        dist = torch.relu(dist + gamma)
        # print(dist.shape)
        dist = dist ** p
        loss_pos = dist.sum() / (y_pred_pos.shape[0] * epoch_pred_neg_samples.shape[0])

    if y_pred_neg.shape[0] > 0:
        dist = y_pred_neg - epoch_pred_pos_samples.unsqueeze(1)  # (B_neg, B_neg)
        # print(dist.shape)
        dist = torch.relu(dist + gamma)
        dist = dist ** p
        loss_pos = dist.sum() / (y_pred_neg.shape[0] * epoch_pred_pos_samples.shape[0])

    return -(loss_pos + loss_neg)


def r2_loss(y_pred, y):
    # all: (B,)
    ss_res = ((y_pred - y) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    loss = 1 - ss_res / ss_tot

    return -loss


def auc_for_reg(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, 1)

    return roc_auc_score(y_true, y_pred)


def neg_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)

    return -np.sqrt(mse)
