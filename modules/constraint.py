#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2021 by Caspar. All rights reserved.
# File Name: constraint.py
# Author: Shankai Yan
# E-mail: dr.skyan@gmail.com
# Created Time: 2021-03-29 17:19:30
###########################################################################
#

import os, sys, pickle

import torch
from torch import nn


class HrchConstraint(nn.Module):
    def __init__(self, num_lbs, hrchrel_path='', binlb={}):
        super(HrchConstraint, self).__init__()
        with open(hrchrel_path, 'rb') as fd:
            hrchrel = pickle.load(fd)
        out_order = binlb.keys()
        if out_order:
            hrchrel = hrchrel[out_order]
            hrchrel = hrchrel.loc[out_order]
        # np.fill_diagonal(hrchrel.values, 1)
        self.hrchrel = torch.tensor(hrchrel.values).float()
        self.linear = nn.Linear(num_lbs, num_lbs)

    def forward(self, logits):
        out = torch.mm(logits, self.linear(self.hrchrel))
        out = out / out.sum(0).expand_as(out)
        out[torch.isnan(out)] = 0
        return logits + out

    def to(self, *args, **kwargs):
        super(HrchConstraint, self).to(*args, **kwargs)
        self.hrchrel = self.hrchrel.to(*args, **kwargs)
        self.linear = self.linear.to(*args, **kwargs)
        return self
