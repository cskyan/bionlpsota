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

from . import reduction as R
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

    def __init__(self, lm_model, config, task_type, embeddim=128, onto_fchdim=128, iactvtn='relu', oactvtn='sigmoid', fchdim=0, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, lm_loss=False, pdrop=0.2, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, do_thrshld=False, constraints=[], initln=False, initln_mean=0., initln_std=0.02, task_params={}, output_layer=-1, pooler=None, layer_pooler='avg', **kwargs):
        config.output_hidden_states = True
        output_layer = list(range(config.num_hidden_layers))
        T.BERTClfHead.__init__(self, lm_model, config, task_type, iactvtn=iactvtn, oactvtn=oactvtn, fchdim=fchdim, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=task_type in ['entlmnt', 'sentsim'] and task_params.setdefault('sentsim_func', None) is not None, lm_loss=lm_loss, pdrop=pdrop, do_norm=do_norm, norm_type=norm_type, do_lastdrop=do_lastdrop, do_crf=do_crf, do_thrshld=do_thrshld, constraints=constraints, initln=initln, initln_mean=initln_mean, initln_std=initln_std, task_params=task_params, output_layer=output_layer, pooler=pooler, layer_pooler=layer_pooler, n_embd=config.hidden_size+int(config.hidden_size/config.num_attention_heads), **kwargs)
        self.num_attention_heads = config.num_attention_heads
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
        self.onto_linear = nn.Sequential(nn.Linear(embeddim, onto_fchdim), self._int_actvtn(), nn.Linear(onto_fchdim, onto_fchdim), self._int_actvtn(), nn.Linear(onto_fchdim, config.num_hidden_layers + config.num_attention_heads), self._out_actvtn())
        if (initln): self.onto_linear.apply(_weights_init(mean=initln_mean, std=initln_std))
        self.halflen = config.num_hidden_layers
        if (type(output_layer) is not int):
            self.output_layer = [x for x in output_layer if (x >= -self.num_hidden_layers and x < self.num_hidden_layers)]
            self.layer_pooler = R.TransformerLayerWeightedReduce(reduction=layer_pooler)
        self.att2spans = nn.Linear(self.maxlen, 2)

    def forward(self, input_ids, pool_idx, *extra_inputs, labels=None, past=None, weights=None, embedding_mode=False, ret_ftspans=False, ret_attention=False):
        use_gpu = next(self.parameters()).is_cuda
        sample_weights = extra_inputs[0] if self.sample_weights and len(extra_inputs) > 0 else None
        if embedding_mode:
            extra_inputs = list(extra_inputs)
            extra_inputs.insert(1 if self.sample_weights else 0, torch.zeros(input_ids.size()[0], dtype=torch.long).to('cuda') if use_gpu else torch.zeros(input_ids.size()[0], dtype=torch.long))
            extra_inputs = tuple(extra_inputs)
        outputs = T.BERTClfHead.forward(self, input_ids, pool_idx, *extra_inputs, labels=labels, past=past, weights=weights, embedding_mode=embedding_mode)
        setattr(self, 'num_attention_heads', 12)
        sys.stdout.flush()
        if embedding_mode: return outputs[:,:int(-self.hdim/(self.num_attention_heads+1))]
        if (labels is None):
            clf_logits, all_attentions, pooled_attentions = outputs
            outputs = clf_logits
        else:
            clf_loss, lm_loss, all_attentions, pooled_attentions = outputs
            outputs = clf_loss, lm_loss
        segment_ids = extra_inputs[4] if self.sample_weights else extra_inputs[3]
        spans = (extra_inputs[6] if len(extra_inputs) > 6 else None) if self.sample_weights else (extra_inputs[5] if len(extra_inputs) > 5 else None)
        if labels is None or spans is not None or ret_ftspans or ret_attention:
            # pos_idx, idn_mask, inv_segment_ids = (torch.arange(segment_ids.size()[1]).to('cuda') if use_gpu else torch.arange(segment_ids.size()[1])), torch.ones_like(segment_ids), 1 - segment_ids
            # mask = pos_idx * idn_mask <= pool_idx.view((-1,1)) * idn_mask
            segment_ids_f, masked_inv_segment_ids = segment_ids.float(), (pool_idx * (1 - segment_ids)).float()
            segment_mask = torch.cat([torch.ger(x, y).unsqueeze(0) for x, y in zip(segment_ids_f, masked_inv_segment_ids)], dim=0) + torch.cat([torch.ger(y, x).unsqueeze(0) for x, y in zip(segment_ids_f, masked_inv_segment_ids)], dim=0)
            masked_pooled_attentions = segment_mask * pooled_attentions
            span_logits = (masked_pooled_attentions + masked_pooled_attentions.permute(0, 2, 1)).sum(-1) * masked_inv_segment_ids
            spans_logits = self.att2spans(masked_pooled_attentions)
            start_logits, end_logits = spans_logits.split(1, dim=-1)
            start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)
        if labels is not None:
            if spans is not None:
                start_positions, end_positions = spans.split(1, dim=-1)
                loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                clf_loss = clf_loss + (start_loss + end_loss) / 2
                if sample_weights is not None: clf_loss *= sample_weights
                outputs = clf_loss, lm_loss
        else:
            if ret_ftspans:
                outputs = (outputs, start_logits, end_logits)
            else:
                span_logits = (masked_pooled_attentions + masked_pooled_attentions.permute(0, 2, 1)).sum(-1) * masked_inv_segment_ids
                outputs = (outputs, span_logits)
        return (outputs + (all_attentions, pooled_attentions, masked_pooled_attentions)) if ret_attention else outputs

    def pool(self, input_ids, pool_idx, clf_h, *extra_inputs):
        onto_ids = extra_inputs[1] if self.sample_weights else extra_inputs[0]
        onto_h = self.onto_linear(self.embedding(onto_ids))
        lyrw_h, mlthead_h = onto_h[:,:self.halflen], onto_h[:,self.halflen:]
        lyr_h = [self.pooler(h, pool_idx) for h in clf_h]
        clf_h = self.layer_pooler(lyr_h, lyrw_h).view(lyr_h[0].size())
        output_size = clf_h.size()
        all_attentions = extra_inputs[5] if self.sample_weights else extra_inputs[4]
        segment_ids = extra_inputs[4] if self.sample_weights else extra_inputs[3]
        pooled_attentions = torch.sum(all_attentions.permute(*((1, 0)+tuple(range(2, len(all_attentions.size()))))) * lyrw_h.view(lyrw_h.size()+(1,)*(len(all_attentions.size())-len(lyrw_h.size()))), dim=1)
        pooled_attentions = torch.sum(pooled_attentions * mlthead_h.view(mlthead_h.size()+(1,)*(len(pooled_attentions.size())-len(mlthead_h.size()))), dim=1)
        return torch.cat([clf_h, torch.sum(clf_h.view(output_size[:-1]+(self.halflen, -1)) * mlthead_h.view(*((onto_ids.size()[0], -1)+(1,)*(len(output_size)-1))), 1)], dim=-1), pooled_attentions

    def transformer(self, input_ids, *extra_inputs, pool_idx=None):
        use_gpu = next(self.parameters()).is_cuda
        segment_ids = extra_inputs[4] if self.sample_weights else extra_inputs[3]
        root = OntoBERTClfHead._PyTorchModuleVertex.from_dict({'module':self.lm_model})
        OntoBERTClfHead.stack_dfs(root, 'modify_config', shared_data={'config':{'output_attentions':True, 'output_hidden_states':True}})
        last_hidden_state, pooled_output, all_encoder_layers, all_attentions = self.lm_model.forward(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=pool_idx)
        all_attentions = torch.cat([att.unsqueeze(0) for att in all_attentions], dim=0)
        return all_encoder_layers[self.output_layer] if type(self.output_layer) is int else [all_encoder_layers[x] for x in self.output_layer], pooled_output, all_attentions
