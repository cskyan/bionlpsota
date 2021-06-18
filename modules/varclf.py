#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2021 by Caspar. All rights reserved.
# File Name: varclf.py
# Author: Shankai Yan
# E-mail: dr.skyan@gmail.com
# Created Time: 2021-03-29 17:19:30
###########################################################################
#

import os, sys, logging

import pandas as pd

import torch
from torch import nn

from . import transformer as T


class MultiClfHead(nn.Module):
    def __init__(self, linears):
        super(MultiClfHead, self).__init__()
        self.linears = linears

    def forward(self, hidden_states):
        return torch.cat([lnr(hidden_states) for lnr in self.linears], dim=-1)

    def to(self, *args, **kwargs):
        super(MultiClfHead, self).to(*args, **kwargs)
        self.linears = [lnr.to(*args, **kwargs) for lnr in self.linears]
        return self


class MultiClfTransformer(object):
    def __init__(self, clf):
        self.clf = clf

    def merge_linear(self, num_linear=-1):
        use_gpu = next(self.clf.parameters()).is_cuda
        self.clf.linear = MultiClfHead(self.clf.linears if num_linear <=0 else self.clf.linears[:num_linear])
        self.clf.linear = self.clf.linear.to('cuda') if use_gpu else self.clf.linear
        self.clf.num_lbs = self.clf._total_num_lbs
        self.clf.binlb = self.clf.global_binlb
        self.clf.binlbr = self.clf.global_binlbr


class OntoBERTClfHead(T.BERTClfHead):
    from bionlp.util.func import DFSVertex, stack_dfs
    class _PyTorchModuleVertex(DFSVertex):
        @property
        def children(self):
            return [OntoBERTClfHead._PyTorchModuleVertex.from_dict({'module':getattr(self.module, attr)}) for attr in dir(self.module) if not attr.startswith('__') and attr != 'base_model' and isinstance(getattr(self.module, attr), nn.Module)] + [OntoBERTClfHead._PyTorchModuleVertex.from_dict({'module':sub_module}) for attr in dir(self.module) if not attr.startswith('__') and attr != 'base_model' and isinstance(getattr(self.module, attr), nn.ModuleList) for sub_module in getattr(self.module, attr)]

        def modify_config(self, shared_data):
            config = shared_data['config']
            for k, v in config.items():
                if hasattr(self.module, k): setattr(self.module, k, v)

    def __init__(self, config, lm_model, lm_config, embeddim=128, onto_fchdim=128, iactvtn='relu', oactvtn='sigmoid', fchdim=0, extfc=False, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, lm_loss=False, do_drop=True, pdrop=0.2, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, do_thrshld=False, constraints=[], initln=False, initln_mean=0., initln_std=0.02, task_params={}, output_layer=-1, pooler=None, layer_pooler='avg', **kwargs):
        from util import func as H
        from . import reduction as R
        lm_config.output_hidden_states = True
        output_layer = list(range(lm_config.num_hidden_layers))
        T.BERTClfHead.__init__(self, config, lm_model, lm_config, iactvtn=iactvtn, oactvtn=oactvtn, fchdim=fchdim, extfc=extfc, sample_weights=sample_weights, num_lbs=num_lbs, lm_loss=lm_loss, do_drop=do_drop, pdrop=pdrop, do_norm=do_norm, norm_type=norm_type, do_lastdrop=do_lastdrop, do_crf=do_crf, do_thrshld=do_thrshld, constraints=constraints, initln=initln, initln_mean=initln_mean, initln_std=initln_std, task_params=task_params, output_layer=output_layer, pooler=pooler, layer_pooler=layer_pooler, n_embd=lm_config.hidden_size+int(lm_config.hidden_size/lm_config.num_attention_heads), **kwargs)
        if hasattr(config, 'onto_df') and type(config.onto_df) is pd.DataFrame:
            self.onto = config.onto_df
        else:
            onto_fpath = config.onto if hasattr(config, 'onto') and os.path.exists(config.onto) else 'onto.csv'
            logging.info('Reading ontology dictionary file [%s]...' % onto_fpath)
            self.onto = pd.read_csv(onto_fpath, sep=kwargs.setdefault('sep', '\t'), index_col='id')
            setattr(config, 'onto_df', self.onto)
        self.embeddim = embeddim
        self.embedding = nn.Embedding(self.onto.shape[0]+1, embeddim)
        self.onto_fchdim = onto_fchdim
        self.onto_linear = nn.Sequential(nn.Linear(embeddim, onto_fchdim), self._int_actvtn(), nn.Linear(onto_fchdim, onto_fchdim), self._int_actvtn(), nn.Linear(onto_fchdim, lm_config.num_hidden_layers + lm_config.num_attention_heads), self._out_actvtn())
        if (initln): self.onto_linear.apply(H._weights_init(mean=initln_mean, std=initln_std))
        self.halflen = lm_config.num_hidden_layers
        if (type(output_layer) is not int):
            self.output_layer = [x for x in output_layer if (x >= -self.num_hidden_layers and x < self.num_hidden_layers)]
            self.layer_pooler = R.TransformerLayerWeightedReduce(reduction=layer_pooler)
        self.att2spans = nn.Linear(self.maxlen, 2)

    def _clf_h(self, hidden_states, mask, all_hidden_states=None, extra_outputs={}):
        # The last element of past is the last_hidden_state
        return (all_hidden_states, torch.stack(mask).max(0)[0]) if type(all_hidden_states) is list else (all_hidden_states, mask)

    def pool(self, input_ids, extra_inputs, mask, clf_h, extra_outputs={}):
        all_clf_h = clf_h
        onto_ids = extra_inputs['onto_id']
        onto_h = self.onto_linear(self.embedding(onto_ids))
        lyrw_h, mlthead_h = onto_h[:,:self.halflen], onto_h[:,self.halflen:]
        if not hasattr(self, 'pooler'): setattr(self, 'pooler', self.__default_pooler__())
        pool_idx = extra_inputs['attention_mask'].sum(1)
        lyr_h = [self.pooler(h, pool_idx) for h in all_clf_h]
        clf_h = self.layer_pooler(lyr_h, lyrw_h).view(lyr_h[0].size())
        output_size = clf_h.size()
        all_attentions = extra_outputs['all_attentions']
        pooled_attentions = torch.sum(all_attentions.permute(*((1, 0)+tuple(range(2, len(all_attentions.size()))))) * lyrw_h.view(lyrw_h.size()+(1,)*(len(all_attentions.size())-len(lyrw_h.size()))), dim=1)
        pooled_attentions = torch.sum(pooled_attentions * mlthead_h.view(mlthead_h.size()+(1,)*(len(pooled_attentions.size())-len(mlthead_h.size()))), dim=1)
        extra_outputs['pooled_attentions'] = pooled_attentions
        return torch.cat([lyr_h[-1], torch.sum(clf_h.view(output_size[:-1]+(self.halflen, -1)) * mlthead_h.view(*((onto_ids.size()[0], -1)+(1,)*(len(output_size)-1))), 1)], dim=-1)

    def transformer(self, input_ids, **extra_inputs):
        # root = OntoBERTClfHead._PyTorchModuleVertex.from_dict({'module':self.lm_model})
        # OntoBERTClfHead.stack_dfs(root, 'modify_config', shared_data={'config':{'output_attentions':True, 'output_hidden_states':True}})
        outputs = self.lm_model(input_ids=input_ids, **dict([(k,v) for k, v in extra_inputs.items() if k != 'onto_id']), return_dict=True, output_attentions=True, output_hidden_states=True)
        outputs['all_attentions'] = torch.cat([att.unsqueeze(0) for att in outputs['attentions']], dim=0)
        outputs['selected_attentions'] = outputs['attentions'][self.output_layer] if type(self.output_layer) is int else [outputs['attentions'][x] for x in self.output_layer]
        return outputs
