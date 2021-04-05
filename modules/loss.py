#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2021 by Caspar. All rights reserved.
# File Name: loss.py
# Author: Shankai Yan
# E-mail: dr.skyan@gmail.com
# Created Time: 2021-03-29 17:19:30
###########################################################################
#

import torch
from torch import nn
import torch.nn.functional as F


class RegressionLoss(nn.Module):
    def __init__(self, reduction='none', x_mode='dist', y_mode='dist'):
        super(RegressionLoss, self).__init__()
        self.reduction = reduction
        self.x_mode, self.y_mode = x_mode, y_mode
        self.ytransform = (lambda x: x) if x_mode == y_mode else (lambda x: 1 - x)

    def forward(self, y_pred, y_true):
        raise NotImplementedError


class MSELoss(RegressionLoss):
    def forward(self, y_pred, y_true):
        return F.mse_loss(y_pred, self.ytransform(y_true), reduction=self.reduction)


class ContrastiveLoss(RegressionLoss):
    def __init__(self, reduction='none', x_mode='dist', y_mode='dist', margin=2.0):
        super(ContrastiveLoss, self).__init__(reduction=reduction, x_mode=x_mode, y_mode=y_mode)
        self.margin = margin

    def forward(self, y_pred, y_true):
        loss = (1 - self.ytransform(y_true)) * torch.pow(y_pred, 2) + self.ytransform(y_true) * torch.pow(torch.clamp(self.margin - y_pred, min=0.0), 2)
        return loss if self.reduction == 'none' else torch.mean(loss)


class HuberLoss(RegressionLoss):
    def forward(self, y_pred, y_true):
        return F.smooth_l1_loss(y_pred, self.ytransform(y_true), reduction=self.reduction)


class LogCoshLoss(RegressionLoss):
    def forward(self, y_pred, y_true):
        ydiff = y_pred - self.ytransform(y_true)
        loss = torch.log(torch.cosh(ydiff + 1e-12))
        return loss if self.reduction == 'none' else torch.mean(loss)


class XTanhLoss(RegressionLoss):
    def forward(self, y_pred, y_true):
        ydiff = y_pred - self.ytransform(y_true)
        loss = ydiff * torch.tanh(ydiff)
        return loss if self.reduction == 'none' else torch.mean(loss)


class XSigmoidLoss(RegressionLoss):
    def forward(self, y_pred, y_true):
        ydiff = y_pred - self.ytransform(y_true)
        loss = 2 * ydiff / (1 + torch.exp(-ydiff)) - ydiff
        return loss if self.reduction == 'none' else torch.mean(loss)

class QuantileLoss(RegressionLoss):
    def __init__(self, reduction='none', x_mode='dist', y_mode='dist', quantile=0.5):
        super(QuantileLoss, self).__init__(reduction=reduction, x_mode=x_mode, y_mode=y_mode)
        self.quantile = 0.5

    def forward(self, y_pred, y_true):
        ydiff = self.ytransform(y_true) - y_pred
        loss = torch.max(self.quantile * ydiff, (self.quantile - 1) * ydiff)
        return loss if self.reduction == 'none' else torch.mean(loss)

class PearsonCorrelationLoss(RegressionLoss):
    def forward(self, y_pred, y_true):
        var_pred = y_pred - torch.mean(y_pred)
        var_true = self.ytransform(y_true) - torch.mean(self.ytransform(y_true))
        # loss = torch.sum(var_pred * var_true) / (torch.sqrt(torch.sum(var_pred ** 2)) * torch.sqrt(torch.sum(var_true ** 2)))
        loss = -1.0 * var_pred * var_true * torch.rsqrt(torch.sum(var_pred ** 2)) * torch.rsqrt(torch.sum(var_true ** 2))
        return loss if self.reduction == 'none' else torch.mean(loss)

