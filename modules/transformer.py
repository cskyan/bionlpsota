#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2021 by Caspar. All rights reserved.
# File Name: transformer.py
# Author: Shankai Yan
# E-mail: dr.skyan@gmail.com
# Created Time: 2021-03-29 17:19:30
###########################################################################
#

import copy

import torch
from torch import nn
import torch.nn.functional as F

from allennlp.modules.conditional_random_field import ConditionalRandomField

from transformers.modeling_bert import BertPreTrainingHeads
from transformers import GPT2LMHeadModel

from util import func as H
from . import reduction as R


class BaseClfHead(nn.Module):
    """ Classifier Head for the Basic Language Model """

    def __init__(self, lm_model, lm_config, config, task_type, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, lm_loss=False, pdrop=0.2, do_norm=True, do_extlin=False, do_lastdrop=True, do_crf=False, do_thrshld=False, constraints=[], task_params={}, binlb={}, binlbr={}, **kwargs):
        super(BaseClfHead, self).__init__()
        self.task_params = task_params
        self.lm_model = lm_model
        self.lm_loss = lm_loss
        self.task_type = task_type
        self.sample_weights = sample_weights
        self.do_norm = do_norm
        self.do_extlin = do_extlin
        self.do_lastdrop = do_lastdrop
        self.dropout = nn.Dropout2d(pdrop) if task_type == 'nmt' else nn.Dropout(pdrop)
        self.last_dropout = nn.Dropout(pdrop) if do_lastdrop else None
        self.crf = ConditionalRandomField(num_lbs) if do_crf else None
        self.thrshlder = R.ThresholdEstimator(last_hdim=kwargs['last_hdim']) if do_thrshld and 'last_hdim' in kwargs else None
        self.thrshld = 0.5
        self.constraints = [cnstrnt_cls(**cnstrnt_params) for cnstrnt_cls, cnstrnt_params in constraints]
        self.mlt_trnsfmr = mlt_trnsfmr # accept multiple streams of inputs, each of which will be input into the transformer
        self.lm_logit = self._mlt_lm_logit if mlt_trnsfmr else self._lm_logit
        self.clf_h = self._clf_h
        self.num_lbs = num_lbs
        self.dim_mulriple = 2 if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] and self.task_params.setdefault('sentsim_func', None) is not None and self.task_params['sentsim_func'] == 'concat' else 1 # two or one sentence
        if self.dim_mulriple > 1 and self.task_params.setdefault('concat_strategy', 'normal') == 'diff': self.dim_mulriple = 4
        self.kwprop = {}
        self.binlb = binlb
        self.global_binlb = copy.deepcopy(binlb)
        self.binlbr = binlbr
        self.global_binlbr = copy.deepcopy(binlbr)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.mode = 'clf'

    def __init_linear__(self):
        raise NotImplementedError

    def forward(self, input_ids, pool_idx, *extra_inputs, labels=None, past=None, weights=None, embedding_mode=False):
        use_gpu = next(self.parameters()).is_cuda
        sample_weights = extra_inputs[0] if self.sample_weights and len(extra_inputs) > 0 else None
        # mask = torch.arange(input_ids.size()[1]).to('cuda').unsqueeze(0).expand(input_ids.size()[:2]) <= pool_idx.unsqueeze(1).expand(input_ids.size()[:2]) if use_gpu else torch.arange(input_ids.size()[1]).unsqueeze(0).expand(input_ids.size()[:2]) <= pool_idx.unsqueeze(1).expand(input_ids.size()[:2])
        # print(('input_ids', [x.size() for x in input_ids] if type(input_ids) is list else input_ids.size()))
        # print(('input_ids', [[','.join(map(str, s.tolist())) for s in x] for x in input_ids] if type(input_ids) is list else [','.join(map(str, s.tolist())) for s in input_ids]))
        # print(('pool_idx', [x.sum(dim=1).float()/x.size()[1] for x in pool_idx] if type(pool_idx) is list else pool_idx.sum(dim=1).float()/pool_idx.size()[1]))
        if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim']:
            trnsfm_output = [self.transformer(input_ids[x], *extra_inputs, pool_idx=pool_idx[x]) for x in [0,1]]
            (hidden_states, past) = zip(*[trnsfm_output[x][:2] if type(trnsfm_output[x]) is tuple else (trnsfm_output[x], None) for x in [0,1]])
            extra_outputs = [trnsfm_output[x][2:] if type(trnsfm_output[x]) is tuple and len(trnsfm_output[x]) > 2 else () for x in [0,1]]
            if len(extra_outputs[0]) == 0: extra_outputs = ()
            hidden_states, past = list(hidden_states), list(past)
        else:
            trnsfm_output = self.transformer(input_ids, *extra_inputs, pool_idx=pool_idx)
            (hidden_states, past) = trnsfm_output[:2] if type(trnsfm_output) is tuple else (trnsfm_output, None)
            extra_outputs = trnsfm_output[2:] if type(trnsfm_output) is tuple and len(trnsfm_output) > 2 else ()
        extra_inputs += extra_outputs
        # print(('after transformer', trnsfm_output))
        if (self.lm_loss):
            lm_logits, lm_target = self.lm_logit(input_ids, hidden_states, *extra_inputs, past=past, pool_idx=pool_idx)
            lm_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            lm_loss = lm_loss_func(lm_logits.contiguous().view(-1, lm_logits.size(-1)), lm_target.contiguous().view(-1)).view(input_ids.size(0), -1)
            if sample_weights is not None: lm_loss *= sample_weights
        else:
            lm_loss = None

        # print(('hdstat: ', [x.size() for x in hidden_states] if type(hidden_states) is list else hidden_states.size()))
        clf_h, pool_idx = self.clf_h(hidden_states, pool_idx, past=past)
        # print(('after clf_h', [x.size() for x in clf_h] if type(clf_h) is list else clf_h.size()))
        pooled_output = self.pool(input_ids, pool_idx, clf_h, *extra_inputs)
        if type(pooled_output) is tuple:
            clf_h = pooled_output[0]
            extra_outputs += pooled_output[1:]
        else:
            clf_h = pooled_output
        # print(('after pool', [x.size() for x in clf_h] if type(clf_h) is list else clf_h.size()))
        if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] and (self.task_params.setdefault('sentsim_func', None) is not None): # default sentsim mode of gpt* is mlt_trnsfmr+_mlt_clf_h
            if self.do_norm: clf_h = [self.norm(clf_h[x]) for x in [0,1]]
            clf_h = [self.dropout(clf_h[x]) for x in [0,1]]
            if self.do_extlin and hasattr(self, 'extlinear'): clf_h = [self.extlinear(clf_h[x]) for x in [0,1]]
            if embedding_mode: return clf_h
            if (self.task_params.setdefault('sentsim_func', None) == 'concat'):
                # clf_h = (torch.cat(clf_h, dim=-1) + torch.cat(clf_h[::-1], dim=-1))
                clf_h = torch.cat(clf_h+[torch.abs(clf_h[0]-clf_h[1]), clf_h[0]*clf_h[1]], dim=-1) if self.task_params.setdefault('concat_strategy', 'normal') == 'diff' else torch.cat(clf_h, dim=-1)
                clf_logits = self.linear(clf_h) if self.linear else clf_h
            elif self.task_type == 'sentsim':
                clf_logits = clf_h = F.pairwise_distance(self.linear(clf_h[0]), self.linear(clf_h[1]), 2, eps=1e-12) if self.task_params['sentsim_func'] == 'dist' else F.cosine_similarity(self.linear(clf_h[0]), self.linear(clf_h[1]), dim=1, eps=1e-12)
        else:
            if self.do_norm: clf_h = self.norm(clf_h)
            # print(('before dropout:', clf_h.size()))
            clf_h = self.dropout(clf_h)
            if self.do_extlin and hasattr(self, 'extlinear'): clf_h = self.extlinear(clf_h)
            if embedding_mode: return clf_h
            # print(('after dropout:', clf_h.size()))
            # print(('linear', self.linear))
            clf_logits = self.linear(clf_h.view(-1, self.n_embd) if self.task_type == 'nmt' else clf_h)
        # print(('after linear:', clf_logits.size()))
        if self.thrshlder: self.thrshld = self.thrshlder(clf_h)
        if self.do_lastdrop: clf_logits = self.last_dropout(clf_logits)
        # print(('after lastdrop:', clf_logits))


        if (labels is None):
            if self.crf:
                tag_seq, score = zip(*self.crf.viterbi_tags(clf_logits.view(input_ids.size()[0], -1, self.num_lbs), torch.ones_like(input_ids)))
                tag_seq = torch.tensor(tag_seq).to('cuda') if use_gpu else torch.tensor(tag_seq)
                print((tag_seq.min(), tag_seq.max(), score))
                clf_logits = torch.zeros((*tag_seq.size(), self.num_lbs)).to('cuda') if use_gpu else torch.zeros((*tag_seq.size(), self.num_lbs))
                clf_logits = clf_logits.scatter(-1, tag_seq.unsqueeze(-1), 1)
                return clf_logits if len(extra_outputs) == 0 else ((clf_logits,) + extra_outputs)
            for cnstrnt in self.constraints: clf_logits = cnstrnt(clf_logits)
            if (self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] and self.task_params.setdefault('sentsim_func', None) is not None and self.task_params['sentsim_func'] != 'concat' and self.task_params['sentsim_func'] != self.task_params.setdefault('ymode', 'sim')): return 1 - clf_logits.view(-1, self.num_lbs)
            return clf_logits.view(-1, self.num_lbs) if len(extra_outputs) == 0 else ((clf_logits.view(-1, self.num_lbs),) + extra_outputs)
        # print((labels.max(), labels.size()))
        if self.crf:
            clf_loss = -self.crf(clf_logits.view(input_ids.size()[0], -1, self.num_lbs), pool_idx)
            if sample_weights is not None: clf_loss *= sample_weights
            return (clf_loss, lm_loss) + extra_outputs
        else:
            for cnstrnt in self.constraints: clf_logits = cnstrnt(clf_logits)
        if self.task_type == 'mltc-clf' or (self.task_type == 'entlmnt' and self.num_lbs > 1) or self.task_type == 'nmt':
            loss_func = nn.CrossEntropyLoss(weight=weights, reduction='none')
            clf_loss = loss_func(clf_logits.view(-1, self.num_lbs), labels.view(-1))
        elif self.task_type == 'mltl-clf' or (self.task_type == 'entlmnt' and self.num_lbs == 1):
            loss_func = nn.BCEWithLogitsLoss(pos_weight=10*weights if weights is not None else None, reduction='none')
            clf_loss = loss_func(clf_logits.view(-1, self.num_lbs), labels.view(-1, self.num_lbs).float())
        elif self.task_type == 'sentsim':
            from util import config as C
            # print((clf_logits.size(), labels.size()))
            loss_cls = C.RGRSN_LOSS_MAP[self.task_params.setdefault('loss', 'contrastive' if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else 'mse')]
            loss_func = loss_cls(reduction='none', x_mode=C.SIM_FUNC_MAP.setdefault(self.task_params['sentsim_func'], 'dist'), y_mode=self.task_params.setdefault('ymode', 'sim')) if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else (loss_cls(reduction='none', x_mode='sim', y_mode=self.task_params.setdefault('ymode', 'sim')) if self.task_params['sentsim_func'] == 'concat' else nn.MSELoss(reduction='none'))
            clf_loss = loss_func(clf_logits.view(-1), labels.view(-1))
        if self.thrshlder:
            num_lbs = labels.view(-1, self.num_lbs).sum(1)
            clf_loss = 0.8 * clf_loss + 0.2 * F.mse_loss(self.thrshld, torch.sigmoid(torch.topk(clf_logits, k=num_lbs.max(), dim=1, sorted=True)[0][:,num_lbs-1]), reduction='mean')
        if sample_weights is not None: clf_loss *= sample_weights
        return (clf_loss, lm_loss) + extra_outputs

    def pool(self, input_ids, pool_idx, clf_h, *extra_inputs):
        use_gpu = next(self.parameters()).is_cuda
        if self.task_type == 'nmt':
            if (hasattr(self, 'layer_pooler')):
                clf_h = self.layer_pooler(clf_h).view(clf_h[0].size())
            else:
                clf_h = clf_h
        elif hasattr(self.lm_model, 'pooler'):
            if self.task_type in ['entlmnt', 'sentsim'] and self.mlt_trnsfmr:
                if (hasattr(self, 'pooler')):
                    if (hasattr(self, 'layer_pooler')):
                        lyr_h = [[self.pooler(h, pool_idx[x]) for h in clf_h[x]] for x in [0,1]]
                        clf_h = [self.layer_pooler(lyr_h[x]).view(lyr_h[x][0].size()) for x in [0,1]]
                    else:
                        clf_h = [self.pooler(clf_h[x], pool_idx[x]) for x in [0,1]]
                else:
                    clf_h = [self.lm_model.pooler(clf_h[x]) for x in [0,1]]
            else:
                if (hasattr(self, 'pooler')):
                    if (hasattr(self, 'layer_pooler')):
                        lyr_h = [self.pooler(h, pool_idx) for h in clf_h]
                        clf_h = self.layer_pooler(lyr_h).view(lyr_h[0].size())
                    else:
                        clf_h = self.pooler(clf_h, pool_idx)
                else:
                    clf_h = self.lm_model.pooler(clf_h)
        else:
            pool_idx = pool_idx.to('cuda') if (use_gpu) else pool_idx
            smp_offset = torch.arange(input_ids[0].size(0)) if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] else torch.arange(input_ids.size(0))
            smp_offset = smp_offset.to('cuda') if use_gpu else smp_offset
            pool_offset = smp_offset * (input_ids[0].size(-1) if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] else input_ids.size(-1)) + pool_idx
            pool_h = pool_offset.unsqueeze(-1).expand(-1, self.n_embd)
            pool_h = pool_h.to('cuda') if use_gpu else pool_h
            clf_h = [clf_h[x].gather(0, pool_h) for x in [0,1]] if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] and (self.task_params.setdefault('sentsim_func', None) is not None) else clf_h.gather(0, pool_h)
        return clf_h

    def _clf_h(self, hidden_states, pool_idx, past=None):
        return ([hidden_states[x].view(-1, self.n_embd) for x in [0,1]], torch.stack(pool_idx).max(0)[0]) if type(hidden_states) is list else (hidden_states.view(-1, self.n_embd), pool_idx)

    def _mlt_clf_h(self, hidden_states, pool_idx, past=None):
        return torch.stack(hidden_states).sum(0).view(-1, self.n_embd), torch.stack(pool_idx).max(0)[0]

    def transformer(self, input_ids, *extra_inputs, pool_idx=None):
        return self.lm_model.transformer(input_ids=input_ids)

    def _lm_logit(self, input_ids, hidden_states, *extra_inputs, past=None, pool_idx=None):
        lm_h = hidden_states[:,:-1]
        return self.lm_model.lm_head(lm_h), input_ids[:,1:]

    def _mlt_lm_logit(self, input_ids, hidden_states, past=None, pool_idx=None):
        lm_h = hidden_states[:,:,:-1].contiguous().view(-1, self.n_embd)
        lm_target = input_ids[:,:,1:].contiguous().view(-1)
        return self.lm_model.lm_head(lm_h), lm_target.view(-1)

    def freeze_lm(self):
        if not hasattr(self, 'lm_model') or self.lm_model is None: return
        for param in self.lm_model.parameters():
            param.requires_grad = False

    def unfreeze_lm(self):
        if not hasattr(self, 'lm_model') or self.lm_model is None: return
        for param in self.lm_model.parameters():
            param.requires_grad = True

    def to(self, *args, **kwargs):
        super(BaseClfHead, self).to(*args, **kwargs)
        self.constraints = [cnstrnt.to(*args, **kwargs) for cnstrnt in self.constraints]
        if hasattr(self, 'linears'): self.linears = [lnr.to(*args, **kwargs) for lnr in self.linears]
        return self

    def add_linear(self, num_lbs, idx=0):
        use_gpu = next(self.parameters()).is_cuda
        self.num_lbs = num_lbs
        self._total_num_lbs = num_lbs if idx==0 else self._total_num_lbs + num_lbs
        self.linear = self.__init_linear__()
        if not hasattr(self, 'linears'): self.linears = []
        self.linears.append(self.linear)

    def _update_global_binlb(self, binlb):
        if not hasattr(self, 'global_binlb'): setattr(self, 'global_binlb', copy.deepcopy(binlb))
        if not hasattr(self, 'global_binlbr'): setattr(self, 'global_binlbr', dict([(v, k) for k, v in binlb.items()]))
        new_lbs = [lb for lb in binlb.keys() if lb not in self.global_binlb]
        self.global_binlb.update(dict([(k, i) for i, k in zip(range(len(self.global_binlb), len(self.global_binlb)+len(new_lbs)), new_lbs)]))
        self.global_binlbr = dict([(v, k) for k, v in self.global_binlb.items()])

    def reset_global_binlb(self):
        delattr(self, 'global_binlb')
        delattr(self, 'global_binlbr')

    def get_linear(self, binlb, idx=0):
        use_gpu = next(self.parameters()).is_cuda
        self.num_lbs = len(binlb)
        self.binlb = binlb
        self.binlbr = dict([(v, k) for k, v in self.binlb.items()])
        self._update_global_binlb(binlb)
        self._total_num_lbs = len(self.global_binlb)
        if not hasattr(self, 'linears'): self.linears = []
        if len(self.linears) <= idx:
            self.linear = self.__init_linear__()
            self.linears.append(self.linear)
            return self.linears[-1]
        else:
            self.linear = self.linears[idx]
            return self.linears[idx]

    def to_siamese(self, from_scratch=False):
        if not hasattr(self, 'clf_task_type') and self.task_type != 'sentsim': self.clf_task_type = self.task_type
        self.task_type = 'sentsim'
        if not hasattr(self, 'clf_num_lbs') and self.task_type != 'sentsim': self.clf_num_lbs = self.num_lbs
        self.num_lbs = 1
        self.mlt_trnsfmr = True if isinstance(self, GPTClfHead) or (isinstance(self, BERTClfHead) and self.task_params.setdefault('sentsim_func', None) is not None) else False
        self.dim_mulriple = 2 if self.task_params.setdefault('sentsim_func', None) == 'concat' else 1
        self.clf_linear = self.linear
        self.linear = self.siamese_linear if hasattr(self, 'siamese_linear') and not from_scratch else self.__init_linear__()
        self.mode = 'siamese'

    def to_clf(self, from_scratch=False):
        self.task_type = self.clf_task_type
        self.num_lbs = self.clf_num_lbs
        if self.mode == 'siamese':
            self.dim_mulriple = 1
            self.siamese_linear = self.linear
        else:
            self.prv_linear = self.linear
        self.linear = self.clf_linear if hasattr(self, 'clf_linear') and not from_scratch else self.__init_linear__()
        self.mode = 'clf'

    def update_params(self, task_params={}, **kwargs):
        self.task_params.update(task_params)
        for k, v in kwargs.items():
            if hasattr(self, k) and type(v) == type(getattr(self, k)):
                if type(v) is dict:
                    getattr(self, k).update(v)
                else:
                    setattr(self, k, v)


class BERTClfHead(BaseClfHead):
    def __init__(self, lm_model, lm_config, config, task_type, iactvtn='relu', oactvtn='sigmoid', fchdim=0, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, lm_loss=False, pdrop=0.2, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, do_thrshld=False, constraints=[], initln=False, initln_mean=0., initln_std=0.02, task_params={}, output_layer=-1, pooler=None, layer_pooler='avg', **kwargs):
        from util import config as C
        super(BERTClfHead, self).__init__(lm_model, lm_config, config, task_type, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=task_type in ['entlmnt', 'sentsim'] and task_params.setdefault('sentsim_func', None) is not None, lm_loss=lm_loss, pdrop=pdrop, do_norm=do_norm, do_lastdrop=do_lastdrop, do_crf=do_crf, do_thrshld=do_thrshld, last_hdim=lm_config.hidden_size, constraints=constraints, task_params=task_params, **kwargs)
        self.lm_head = BertPreTrainingHeads(config)
        self.vocab_size = lm_config.vocab_size
        self.num_hidden_layers = lm_config.num_hidden_layers
        self.n_embd = kwargs.setdefault('n_embd', lm_config.hidden_size)
        self.maxlen = self.task_params.setdefault('maxlen', 128)
        self.norm = C.NORM_TYPE_MAP[norm_type](self.maxlen) if self.task_type == 'nmt' else C.NORM_TYPE_MAP[norm_type](self.n_embd)
        self._int_actvtn = C.ACTVTN_MAP[iactvtn]
        self._out_actvtn = C.ACTVTN_MAP[oactvtn]
        self.fchdim = fchdim
        self.hdim = self.dim_mulriple * self.n_embd if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] else self.n_embd
        # self.linear = nn.Sequential(nn.Linear(self.hdim, self.num_lbs), nn.Sigmoid()) if self.task_type == 'sentsim' else nn.Linear(self.hdim, self.num_lbs)
        self.linear = self.__init_linear__()
        if (initln): self.linear.apply(H._weights_init(mean=initln_mean, std=initln_std))
        # if (initln): self.lm_model.apply(self.lm_model.init_bert_weights)
        if self.do_extlin:
            self.extlinear = nn.Sequential(nn.Linear(self.n_embd, self.n_embd), C.ACTVTN_MAP['tanh']())
            if (initln): self.extlinear.apply(H._weights_init(mean=initln_mean, std=initln_std))
        if (type(output_layer) is int):
            self.output_layer = output_layer if (output_layer >= -self.num_hidden_layers and output_layer < self.num_hidden_layers) else -1
        else:
            self.output_layer = [x for x in output_layer if (x >= -self.num_hidden_layers and x < self.num_hidden_layers)]
            self.layer_pooler = R.TransformerLayerMaxPool(kernel_size=len(self.output_layer)) if layer_pooler == 'max' else R.TransformerLayerAvgPool(kernel_size=len(self.output_layer))
        if pooler is not None:
            self.pooler = R.MaskedReduction(reduction=pooler, dim=1)

    def __init_linear__(self):
        use_gpu = next(self.parameters()).is_cuda
        linear = (nn.Sequential(nn.Linear(self.hdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), *([] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.fchdim, self.num_lbs), self._out_actvtn()])) if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] else nn.Sequential(nn.Linear(self.hdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.num_lbs))) if self.fchdim else (nn.Sequential(*([nn.Linear(self.hdim, self.hdim), self._int_actvtn()] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.hdim, self.num_lbs), self._out_actvtn()])) if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] else nn.Linear(self.hdim, self.num_lbs))
        return linear.to('cuda') if use_gpu else linear


    def _clf_h(self, hidden_states, pool_idx, past=None):
        return hidden_states, pool_idx

    def transformer(self, input_ids, *extra_inputs, pool_idx=None):
        use_gpu = next(self.parameters()).is_cuda
        # mask = torch.arange(input_ids.size()[1]).to('cuda').unsqueeze(0).expand(input_ids.size()[:2]) <= pool_idx.unsqueeze(1).expand(input_ids.size()[:2]) if use_gpu else torch.arange(input_ids.size()[1]).unsqueeze(0).expand(input_ids.size()[:2]) <= pool_idx.unsqueeze(1).expand(input_ids.size()[:2])
        segment_ids = extra_inputs[3] if self.sample_weights else extra_inputs[2]
        if (self.output_layer == -1 or self.output_layer == 11):
            self.lm_model.encoder.output_hidden_states = False
            last_hidden_state, pooled_output = self.lm_model.forward(input_ids=input_ids, token_type_ids=segment_ids.long(), attention_mask=pool_idx)
            return last_hidden_state, pooled_output
        else:
            self.lm_model.encoder.output_hidden_states = True
            outputs = self.lm_model.forward(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=pool_idx)
            last_hidden_state, pooled_output, all_encoder_layers = outputs[:3]
            return all_encoder_layers[self.output_layer] if type(self.output_layer) is int else [all_encoder_layers[x] for x in self.output_layer], pooled_output


    def _lm_logit(self, input_ids, hidden_states, *extra_inputs, past=None, pool_idx=None):
        masked_lm_ids = (extra_inputs[1] if len(extra_inputs) > 1 else input_ids) if self.sample_weights else (extra_inputs[0] if len(extra_inputs) > 0 else input_ids)
        return self.lm_head(*self.transformer(masked_lm_ids, *extra_inputs, pool_idx=pool_idx))[0], masked_lm_lbs


class GPTClfHead(BaseClfHead):
    def __init__(self, lm_model, lm_config, config, task_type, iactvtn='relu', oactvtn='sigmoid', fchdim=0, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, lm_loss=False, pdrop=0.2, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, initln=False, initln_mean=0., initln_std=0.02, task_params={}, **kwargs):
        from util import config as C
        super(GPTClfHead, self).__init__(lm_model, lm_config, config, task_type, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=task_type == 'sentsim', lm_loss=lm_loss, pdrop=pdrop, do_norm=do_norm, do_lastdrop=do_lastdrop, do_crf=do_crf, task_params=task_params, **kwargs)
        if type(lm_model) is GPT2LMHeadModel:self.kwprop['past_paramname'] = 'past'
        self.vocab_size = lm_config.vocab_size
        self.n_embd = lm_config.n_embd
        self.norm = C.NORM_TYPE_MAP[norm_type](lm_config.n_embd)
        self.clf_h = self._mlt_clf_h if self.task_type in ['entlmnt', 'sentsim'] and self.mlt_trnsfmr and (task_params.setdefault('sentsim_func', None) is None) else self._clf_h
        self._int_actvtn = C.ACTVTN_MAP[iactvtn]
        self._out_actvtn = C.ACTVTN_MAP[oactvtn]
        self.fchdim = fchdim
        self.hdim = self.dim_mulriple * self.n_embd if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] else self.n_embd
        self.linear = self.__init_linear__()
        if (initln): self.linear.apply(H._weights_init(mean=initln_mean, std=initln_std))
        if self.do_extlin:
            self.extlinear = nn.Linear(self.n_embd, self.n_embd)
            if (initln): self.extlinear.apply(H._weights_init(mean=initln_mean, std=initln_std))

    def __init_linear__(self):
        return nn.Sequential(nn.Linear(self.hdim, self.num_lbs), nn.Sigmoid()) if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] else nn.Linear(self.hdim, self.num_lbs)


class TransformXLClfHead(BaseClfHead):
    def __init__(self, lm_model, lm_config, config, task_type, iactvtn='relu', oactvtn='sigmoid', fchdim=0, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, lm_loss=False, pdrop=0.2, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, initln=False, initln_mean=0., initln_std=0.02, task_params={}, **kwargs):
        from util import config as C
        super(TransformXLClfHead, self).__init__(lm_model, lm_config, config, task_type, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=mlt_trnsfmr, lm_loss=lm_loss, pdrop=pdrop, do_norm=do_norm, do_lastdrop=do_lastdrop, do_crf=do_crf, task_params=task_params, **kwargs)
        self.kwprop['past_paramname'] = 'mems'
        self.vocab_size = lm_config.n_token
        self.n_embd = lm_config.d_embed
        self.norm = C.NORM_TYPE_MAP[norm_type](lm_config.n_embd)
        self._int_actvtn = C.ACTVTN_MAP[iactvtn]
        self._out_actvtn = C.ACTVTN_MAP[oactvtn]
        self.fchdim = fchdim
        self.hdim = self.dim_mulriple * self.n_embd if self.mlt_trnsfmr and self.task_type in ['sentsim'] else self.n_embd
        self.linear = self.__init_linear__()
        if (initln): self.linear.apply(H._weights_init(mean=initln_mean, std=initln_std))

    def __init_linear__(self):
        return nn.Linear(self.hdim, self.num_lbs)

    def _lm_logit(self, input_ids, hidden_states, *extra_inputs, past=None, pool_idx=None):
        bsz, tgt_len = input_ids.size(0), input_ids.size(1)
        lm_h, lm_trgt = hidden_states[:, -tgt_len:], None
        if self.lm_model.sample_softmax > 0 and self.lm_model.training:
            assert self.lm_model.config.tie_weight
            logit = sample_logits(self.lm_model.transformer.word_emb, self.lm_model.out_layer.bias, lm_trgt, lm_h, self.lm_model.sampler)
            softmax_output = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            softmax_output = self.lm_model.crit(lm_h.contiguous().view(-1, lm_h.size(-1)), lm_trgt)
            if lm_trgt is None:
                softmax_output = softmax_output.view(bsz, tgt_len, -1)
            else:
                softmax_output = softmax_output.view(bsz, tgt_len)
        return softmax_output, input_ids

    def _mlt_lm_logit(self, input_ids, hidden_states, *extra_inputs, past=None, pool_idx=None):
        return self._lm_logit(input_ids, hidden_states, *extra_inputs, past=past, pool_idx=pool_idx)

    def transformer(self, input_ids, *extra_inputs, pool_idx=None):
        return self.lm_model.transformer(input_ids=input_ids.view(input_ids.size(0), -1))
