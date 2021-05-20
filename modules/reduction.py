#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2021 by Caspar. All rights reserved.
# File Name: reduction.py
# Author: Shankai Yan
# E-mail: dr.skyan@gmail.com
# Created Time: 2021-03-29 17:19:30
###########################################################################
#

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from util import config as C


class TransformerLayerMaxPool(nn.MaxPool1d):
    def forward(self, hidden_states):
        if type(hidden_states) is list:
            hidden_states = torch.cat([x.unsqueeze(0) for x in hidden_states], dim=0)
        hidden_size = hidden_states.size()
        lyr, bs = hidden_size[:2]
        hidden_states = hidden_states.view(lyr, bs, np.prod(hidden_size[2:])).permute(1, 2, 0)
        pooled_hidden_states = F.max_pool1d(hidden_states, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, self.return_indices)
        _, _, plyr = pooled_hidden_states.size()
        pooled_hidden_states = pooled_hidden_states.permute(2, 0, 1)
        return pooled_hidden_states.view(plyr, bs, *hidden_size[2:])


class TransformerLayerAvgPool(nn.AvgPool1d):
    def forward(self, hidden_states):
        if type(hidden_states) is list:
            hidden_states = torch.cat([x.unsqueeze(0) for x in hidden_states], dim=0)
        hidden_size = hidden_states.size()
        lyr, bs = hidden_size[:2]
        hidden_states = hidden_states.view(lyr, bs, np.prod(hidden_size[2:])).permute(1, 2, 0)
        pooled_hidden_states = F.avg_pool1d(hidden_states, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
        _, _, plyr = pooled_hidden_states.size()
        pooled_hidden_states = pooled_hidden_states.permute(2, 0, 1)
        return pooled_hidden_states.view(plyr, bs, *hidden_size[2:])


class TransformerLayerWeightedReduce(nn.Module):
    def __init__(self, reduction='sum'):
        super(TransformerLayerWeightedReduce, self).__init__()
        self.reduction = reduction

    def forward(self, hidden_states, *extra_inputs):
        if type(hidden_states) is list:
            hidden_states = torch.cat([x.unsqueeze(0) for x in hidden_states], dim=0)
        weights = extra_inputs[0]
        hidden_size, weight_size = hidden_states.size(), weights.size()
        lyr, bs = hidden_size[:2]
        w = (torch.cat((weights, torch.zeros((weight_size[0], lyr-weight_size[1]))), dim=1) if lyr > weight_size[1] else weights[:,:lyr]).permute(1, 0).view((lyr, -1)+(1,)*(len(hidden_size)-2))
        wsum_hidden_states = torch.sum(hidden_states * w, 0) if self.reduction == 'sum' else torch.mean(hidden_states * w, 0)
        return wsum_hidden_states.view(bs, *hidden_size[2:])


class MaskedReduction(nn.Module):
    def __init__(self, reduction='mean', dim=-1):
        super(MaskedReduction, self).__init__()
        self.reduction = reduction
        self.dim = dim

    def forward(self, hidden_states, mask):
        if type(hidden_states) is list and len(hidden_states) == 2:
            return [self._forward(hidden_states[x], mask[x]) for x in [0,1]]
        else:
            return self._forward(hidden_states, mask)


    def _forward(self, hidden_states, mask):
        hidden_states_size, mask_size = hidden_states.size(), mask.size()
        while True:
            try:
            	sq_idx = mask_size.index(1)
            	mask = mask.squeeze(sq_idx)
            	mask_size = mask.size()
            except Exception as e:
            	break
        missed_dim = hidden_states_size[len(mask_size):]
        for x, d in zip(missed_dim, range(len(mask_size), len(hidden_states_size))):
            mask = mask.unsqueeze(-1).expand(*([-1] * d), x)
        if type(self.dim) is list:
            for d in self.dim:
                if self.reduction == 'mean':
                    hidden_states = (hidden_states * mask.float()).mean(dim=d, keepdim=True)
                elif self.reduction == 'sum':
                    hidden_states = (hidden_states * mask.float()).sum(dim=d, keepdim=True)
                elif self.reduction == 'max':
                    hidden_states = (hidden_states * mask.float()).max(dim=d, keepdim=True)[0]
                else:
                    hidden_states = hidden_states[:,0]
                mask = mask.mean(dim=d, keepdim=True)
            return hidden_states.view(*[hidden_states_size[x] for x in range(len(hidden_states_size)) if x not in self.dim])
        else:
            if self.reduction == 'mean':
                return (hidden_states * mask.float()).mean(dim=self.dim)
            elif self.reduction == 'sum':
                return (hidden_states * mask.float()).sum(dim=self.dim)
            elif self.reduction == 'max':
                return (hidden_states * mask.float()).max(dim=self.dim)[0]
            else:
                return hidden_states[:,0]


class ThresholdEstimator(nn.Module):
    def __init__(self, last_hdim, fchdim=100, iactvtn='relu', oactvtn='sigmoid', init_thrshld=0.5):
        super(ThresholdEstimator, self).__init__()
        self.thrshld = init_thrshld
        self.linear = nn.Sequential(nn.Linear(last_hdim, fchdim), C.ACTVTN_MAP[iactvtn](), nn.Linear(fchdim, fchdim), C.ACTVTN_MAP[iactvtn](), nn.Linear(fchdim, 1), C.ACTVTN_MAP[oactvtn]()) if fchdim else nn.Sequential(nn.Linear(last_hdim, 1), C.ACTVTN_MAP[oactvtn]())

    def forward(self, logits):
        return self.linear(logits)
