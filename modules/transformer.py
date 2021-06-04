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

import copy, logging

import torch
from torch import nn
import torch.nn.functional as F

from allennlp.modules.conditional_random_field import ConditionalRandomField

from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class BaseClfHead(nn.Module):
    """ All-in-one Classifier Head for the Basic Language Model """

    def __init__(self, config, lm_model, lm_config, num_lbs=1, mlt_trnsfmr=False, task_params={}, binlb={}, binlbr={}, **kwargs):
        from . import reduction as R
        super(BaseClfHead, self).__init__()
        self.lm_model = lm_model
        self.lm_config = lm_config
        self.input_keys = config.input_keys
        self.maxlen = config.maxlen
        self.lm_loss = kwargs.setdefault('lm_loss', config.lm_loss if hasattr(config, 'lm_loss') else True)
        self.lm_head = self.__lm_head__()
        self.num_lbs = num_lbs
        pdrop = kwargs.setdefault('pdrop', config.pdrop if hasattr(config, 'pdrop') else 0.2)
        self.sample_weights = kwargs.setdefault('sample_weights', config.lm_loss if hasattr(config, 'sample_weights') else False)
        self.mlt_trnsfmr = mlt_trnsfmr # accept multiple streams of inputs, each of which will be input into the transformer
        self.task_type = kwargs.setdefault('task_type', config.task_type)
        self.task_params = task_params

        self.do_norm = kwargs.setdefault('do_norm', config.do_norm if hasattr(config, 'do_norm') else False)
        self.do_extlin = kwargs.setdefault('do_extlin', config.do_extlin if hasattr(config, 'do_extlin') else True)
        self.do_lastdrop = kwargs.setdefault('do_lastdrop', config.do_lastdrop if hasattr(config, 'do_lastdrop') else True)
        self.dropout = nn.Dropout2d(pdrop) if self.task_type == 'nmt' else nn.Dropout(pdrop)
        self.last_dropout = nn.Dropout(pdrop) if self.do_lastdrop else None
        do_crf = kwargs.setdefault('do_crf', config.do_crf if hasattr(config, 'do_crf') else False)
        self.crf = ConditionalRandomField(num_lbs) if do_crf else None
        constraints = kwargs.setdefault('cnstrnts', config.cnstrnts.split(',') if hasattr(config, 'cnstrnts') and config.cnstrnts else [])
        self.constraints = [cnstrnt_cls(**cnstrnt_params) for cnstrnt_cls, cnstrnt_params in constraints]
        do_thrshld = kwargs.setdefault('do_thrshld', config.do_thrshld if hasattr(config, 'do_thrshld') else False)
        self.thrshlder = R.ThresholdEstimator(last_hdim=kwargs['last_hdim']) if do_thrshld and 'last_hdim' in kwargs else None
        self.thrshld = kwargs.setdefault('thrshld', 0.5)

        # Customerized function calling
        self.lm_logit = self._mlt_lm_logit if self.mlt_trnsfmr else self._lm_logit
        self.clf_h = self._clf_h
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
        self.debug = config.verbose if hasattr(config, 'verbose') else False

    def __init_linear__(self):
        raise NotImplementedError

    def __lm_head__(self):
        raise NotImplementedError

    def __default_pooler__(self):
        from . import reduction as R
        return R.MaskedReduction(reduction=None, dim=1)

    def forward(self, input_ids, *extra_inputs, labels=None, all_hidden_states=None, weights=None, embedding_mode=False):
        use_gpu = next(self.parameters()).is_cuda
        if self.sample_weights and len(extra_inputs) > 0:
            sample_weights = extra_inputs[-1]
            extra_inputs = extra_inputs[:-1]
        else:
            sample_weights = None
        extra_inputs_dict = dict(zip([x for x in self.input_keys if x != 'input_ids'], extra_inputs))
        pool_idx = extra_inputs_dict['attention_mask'].sum(1)
        mask = extra_inputs_dict['attention_mask']
        if self.debug:
            logging.debug(('size of input_ids', [x.size() for x in input_ids] if type(input_ids) is list else input_ids.size()))
            logging.debug(('input_ids', [[','.join(map(str, s.tolist())) for s in x] for x in input_ids[:5]] if type(input_ids) is list else [','.join(map(str, s.tolist())) for s in input_ids[:5]]))
        # Go through the language model
        output_fields = set(['last_hidden_state', 'hidden_states'])
        if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim']:
            trnsfm_output = [self.transformer(input_ids[x], **extra_inputs_dict) for x in [0,1]]
            hidden_states, all_hidden_states = zip(*[[trnsfm_output[x][k] if k in trnsfm_output[x] else None for k in ['last_hidden_state', 'hidden_states']] for x in [0,1]])
            hidden_states, all_hidden_states = list(hidden_states), list(all_hidden_states)
            extra_outputs = [dict([(k, v) for k, v in trnsfm_output[x].items() if k not in output_fields]) for x in [0,1]]
        else:
            trnsfm_output = self.transformer(input_ids, **extra_inputs_dict)
            hidden_states, all_hidden_states = (trnsfm_output[k] if k in trnsfm_output else None for k in ['last_hidden_state', 'hidden_states'])
            extra_outputs = dict([(k, v) for k, v in trnsfm_output.items() if k not in output_fields])
        if self.debug: logging.debug(('after transformer', trnsfm_output[:5]))

        # Calculate language model loss
        if (self.lm_loss):
            lm_logits, lm_target = self.lm_logit(input_ids, extra_inputs_dict, hidden_states, all_hidden_states=all_hidden_states, extra_outputs=extra_outputs)
            lm_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            lm_loss = lm_loss_func(lm_logits.contiguous().view(-1, lm_logits.size(-1)), lm_target.contiguous().view(-1)).view(input_ids.size(0), -1)
            if sample_weights is not None: lm_loss *= sample_weights
        else:
            lm_loss = None

        # Pooling
        if self.debug: logging.debug(('hdstat: ', [x.size() for x in hidden_states] if type(hidden_states) is list else hidden_states.size()))
        clf_h, mask = self.clf_h(hidden_states, mask, all_hidden_states=all_hidden_states, extra_outputs=extra_outputs)
        if self.debug: logging.debug(('after clf_h', [x.size() for x in clf_h] if type(clf_h) is list else clf_h.size()))
        clf_h = self.pool(input_ids, extra_inputs_dict, mask, clf_h, extra_outputs=extra_outputs)
        if self.debug: logging.debug(('after pool', [x.size() for x in clf_h] if type(clf_h) is list else clf_h.size()))

        # Other layers
        if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] and (self.task_params.setdefault('sentsim_func', None) is not None): # default sentsim mode of gpt* is mlt_trnsfmr+_mlt_clf_h
            if self.do_norm: clf_h = [self.norm(clf_h[x]) for x in [0,1]]
            if self.do_drop: clf_h = [self.dropout(clf_h[x]) for x in [0,1]]
            if self.do_extlin and hasattr(self, 'extlinear'): clf_h = [self.extlinear(clf_h[x]) for x in [0,1]]
            if embedding_mode: return clf_h
            if self.task_params.setdefault('sentsim_func', None) == 'concat':
                if self.task_params.setdefault('concat_strategy', 'normal') == 'reverse':
                    clf_h = (torch.cat(clf_h, dim=-1) + torch.cat(clf_h[::-1], dim=-1))
                elif self.task_params['concat_strategy'] == 'diff':
                    clf_h = torch.cat(clf_h+[torch.abs(clf_h[0]-clf_h[1]), clf_h[0]*clf_h[1]], dim=-1)
                else:
                    clf_h = torch.cat(clf_h, dim=-1)
                clf_logits = self.linear(clf_h) if self.linear else clf_h
            elif self.task_type == 'sentsim':
                clf_logits = clf_h = F.pairwise_distance(self.linear(clf_h[0]), self.linear(clf_h[1]), 2, eps=1e-12) if self.task_params['sentsim_func'] == 'dist' else F.cosine_similarity(self.linear(clf_h[0]), self.linear(clf_h[1]), dim=1, eps=1e-12)
        else:
            if self.do_norm: clf_h = self.norm(clf_h)
            if self.debug: logging.debug(('before dropout:', clf_h.size()))
            if self.do_drop: clf_h = self.dropout(clf_h)
            if self.do_extlin and hasattr(self, 'extlinear'): clf_h = self.extlinear(clf_h)
            if embedding_mode: return clf_h
            if self.debug: logging.debug(('after dropout:', clf_h.size()))
            if self.debug: logging.debug(('linear', self.linear))
            clf_logits = self.linear(clf_h.view(-1, self.n_embd) if self.task_type == 'nmt' else clf_h)
        if self.debug: logging.debug(('after linear:', clf_logits.size()))
        if self.thrshlder: self.thrshld = self.thrshlder(clf_h)
        if self.do_lastdrop: clf_logits = self.last_dropout(clf_logits)
        if self.debug: logging.debug(('after lastdrop:', clf_logits[:5]))

        if (labels is None):
            if self.crf:
                tag_seq, score = zip(*self.crf.viterbi_tags(clf_logits.view(input_ids.size()[0], -1, self.num_lbs), torch.ones_like(input_ids)))
                tag_seq = torch.tensor(tag_seq).to('cuda') if use_gpu else torch.tensor(tag_seq)
                if self.debug: logging.debug((tag_seq.min(), tag_seq.max(), score))
                clf_logits = torch.zeros((*tag_seq.size(), self.num_lbs)).to('cuda') if use_gpu else torch.zeros((*tag_seq.size(), self.num_lbs))
                clf_logits = clf_logits.scatter(-1, tag_seq.unsqueeze(-1), 1)
                return clf_logits, extra_outputs
            for cnstrnt in self.constraints: clf_logits = cnstrnt(clf_logits)
            if (self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] and self.task_params.setdefault('sentsim_func', None) is not None and self.task_params['sentsim_func'] != 'concat' and self.task_params['sentsim_func'] != self.task_params.setdefault('ymode', 'sim')): return 1 - clf_logits.view(-1, self.num_lbs)
            return clf_logits.view(-1, self.num_lbs), extra_outputs
        if self.debug: logging.debug(('label max: ', labels.max(), 'label size: ', labels.size()))
        if self.crf:
            clf_loss = -self.crf(clf_logits.view(input_ids.size()[0], -1, self.num_lbs), pool_idx)
            if sample_weights is not None: clf_loss *= sample_weights
            return clf_loss, lm_loss, extra_outputs
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
            if self.debug: logging.debug(('clf logits: ', clf_logits.size()))
            loss_cls = C.RGRSN_LOSS_MAP[self.task_params.setdefault('loss', 'contrastive' if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else 'mse')]
            loss_func = loss_cls(reduction='none', x_mode=C.SIM_FUNC_MAP.setdefault(self.task_params['sentsim_func'], 'dist'), y_mode=self.task_params.setdefault('ymode', 'sim')) if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else (loss_cls(reduction='none', x_mode='sim', y_mode=self.task_params.setdefault('ymode', 'sim')) if self.task_params['sentsim_func'] == 'concat' else nn.MSELoss(reduction='none'))
            clf_loss = loss_func(clf_logits.view(-1), labels.view(-1))
        if self.thrshlder:
            num_lbs = labels.view(-1, self.num_lbs).sum(1)
            clf_loss = 0.8 * clf_loss + 0.2 * F.mse_loss(self.thrshld, torch.sigmoid(torch.topk(clf_logits, k=num_lbs.max(), dim=1, sorted=True)[0][:,num_lbs-1]), reduction='mean')
        if sample_weights is not None: clf_loss *= sample_weights
        return clf_loss, lm_loss, extra_outputs

    def pool(self, input_ids, extra_inputs, mask, clf_h, extra_outputs={}):
        use_gpu = next(self.parameters()).is_cuda
        if self.task_type == 'nmt':
            if (hasattr(self, 'layer_pooler')):
                clf_h = self.layer_pooler(clf_h)
            else:
                clf_h = clf_h
        else:
            if not hasattr(self, 'pooler'): setattr(self, 'pooler', self.__default_pooler__())
            if self.task_type in ['entlmnt', 'sentsim'] and self.mlt_trnsfmr:
                if (hasattr(self, 'layer_pooler')):
                    lyr_h = [[self.pooler(h, mask[x]) for h in clf_h[x]] for x in [0,1]]
                    clf_h = [self.layer_pooler(lyr_h[x]) for x in [0,1]]
                else:
                    clf_h = [self.pooler(clf_h[x], mask[x]) for x in [0,1]]
            else:
                if (hasattr(self, 'layer_pooler')):
                    lyr_h = [self.pooler(h, mask) for h in clf_h]
                    clf_h = self.layer_pooler(lyr_h)
                else:
                    clf_h = self.pooler(clf_h, mask)
        return clf_h

    def _clf_h(self, hidden_states, mask, all_hidden_states=None, extra_outputs={}):
        return (hidden_states, torch.stack(mask).max(0)[0]) if type(hidden_states) is list else (hidden_states, mask)

    def _mlt_clf_h(self, hidden_states, mask, all_hidden_states=None, extra_outputs={}):
        return torch.stack(hidden_states).sum(0), torch.stack(pool_idx).max(0)[0]

    def transformer(self, input_ids, **extra_inputs):
        return self.lm_model(input_ids=input_ids, **extra_inputs, return_dict=True)

    def _lm_logit(self, input_ids, extra_inputs, hidden_states, all_hidden_states=None, extra_outputs={}):
        lm_h = hidden_states[:,:-1]
        return self.lm_head(lm_h), input_ids[:,1:]

    def _mlt_lm_logit(self, input_ids, hidden_states, extra_inputs, all_hidden_states=None, extra_outputs={}):
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
    def __init__(self, config, lm_model, lm_config, iactvtn='relu', oactvtn='sigmoid', fchdim=0, extfc=False, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, lm_loss=False, do_drop=True, pdrop=0.2, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, do_thrshld=False, constraints=[], initln=False, initln_mean=0., initln_std=0.02, task_params={}, output_layer=-1, pooler=None, layer_pooler='avg', **kwargs):
        from util import config as C
        from util import func as H
        from . import reduction as R
        super(BERTClfHead, self).__init__(config, lm_model, lm_config, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=config.task_type in ['entlmnt', 'sentsim'] and task_params.setdefault('sentsim_func', None) is not None, lm_loss=lm_loss, do_drop=do_drop, pdrop=pdrop, do_norm=do_norm, do_lastdrop=do_lastdrop, do_crf=do_crf, do_thrshld=do_thrshld, last_hdim=lm_config.hidden_size, constraints=constraints, task_params=task_params, **kwargs)
        self.vocab_size = lm_config.vocab_size
        self.num_hidden_layers = lm_config.num_hidden_layers
        self.n_embd = kwargs.setdefault('n_embd', lm_config.hidden_size)
        self.norm = C.NORM_TYPE_MAP[norm_type](self.maxlen) if self.task_type == 'nmt' else C.NORM_TYPE_MAP[norm_type](self.n_embd)
        self._int_actvtn = C.ACTVTN_MAP[iactvtn]
        self._out_actvtn = C.ACTVTN_MAP[oactvtn]
        self.fchdim = fchdim
        self.extfc = extfc
        self.hdim = self.dim_mulriple * self.n_embd if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] else self.n_embd
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
        linear = (nn.Sequential(nn.Linear(self.hdim, self.fchdim), self._int_actvtn(), *([nn.Linear(self.fchdim, self.fchdim), self._int_actvtn()] if self.extfc else []), *([] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.fchdim, self.num_lbs), self._out_actvtn()])) if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] else nn.Sequential(nn.Linear(self.hdim, self.fchdim), self._int_actvtn(), *([nn.Linear(self.fchdim, self.fchdim), self._int_actvtn()] if self.extfc else []), nn.Linear(self.fchdim, self.num_lbs))) if self.fchdim else (nn.Sequential(*([nn.Linear(self.hdim, self.hdim), self._int_actvtn()] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.hdim, self.num_lbs), self._out_actvtn()])) if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] else nn.Linear(self.hdim, self.num_lbs))
        return linear.to('cuda') if use_gpu else linear

    def __lm_head__(self):
        return BertOnlyMLMHead(self.lm_config)

    def _clf_h(self, hidden_states, mask, all_hidden_states=None, extra_outputs={}):
        return hidden_states, mask

    def transformer(self, input_ids, **extra_inputs):
        if (self.output_layer == -1 or self.output_layer == 11):
            self.lm_model.encoder.output_hidden_states = False
            return self.lm_model.forward(input_ids=input_ids, **extra_inputs, return_dict=True)
        else:
            self.lm_model.encoder.output_hidden_states = True
            outputs = self.lm_model.forward(input_ids=input_ids, **extra_inputs, return_dict=True)
            all_encoder_layers = outputs['hidden_states']
            outputs['hidden_states'] = all_encoder_layers[self.output_layer] if type(self.output_layer) is int else [all_encoder_layers[x] for x in self.output_layer]
            return outputs


    def _lm_logit(self, input_ids, hidden_states, extra_inputs, all_hidden_states=None):
        masked_lm_ids = (extra_inputs[1] if len(extra_inputs) > 1 else input_ids) if self.sample_weights else (extra_inputs[0] if len(extra_inputs) > 0 else input_ids)
        return self.lm_head(*self.transformer(masked_lm_ids, *extra_inputs, pool_idx=pool_idx))[0], masked_lm_lbs


class GPTClfHead(BaseClfHead):
    def __init__(self, config, lm_model, lm_config, iactvtn='relu', oactvtn='sigmoid', fchdim=0, extfc=False, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, lm_loss=False, do_drop=False, pdrop=0.2, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, initln=False, initln_mean=0., initln_std=0.02, task_params={}, **kwargs):
        from util import config as C
        from util import func as H
        super(GPTClfHead, self).__init__(config, lm_model, lm_config, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=config.task_type == 'sentsim', lm_loss=lm_loss, do_drop=do_drop, pdrop=pdrop, do_norm=do_norm, do_lastdrop=do_lastdrop, do_crf=do_crf, task_params=task_params, **kwargs)
        from transformers import GPT2Model
        self.vocab_size = lm_config.vocab_size
        self.n_embd = lm_config.n_embd
        self.norm = C.NORM_TYPE_MAP[norm_type](lm_config.n_embd)
        self.clf_h = self._mlt_clf_h if self.task_type in ['entlmnt', 'sentsim'] and self.mlt_trnsfmr and (task_params.setdefault('sentsim_func', None) is None) else self._clf_h
        self._int_actvtn = C.ACTVTN_MAP[iactvtn]
        self._out_actvtn = C.ACTVTN_MAP[oactvtn]
        self.fchdim = fchdim
        self.extfc = extfc
        self.hdim = self.dim_mulriple * self.n_embd if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] else self.n_embd
        self.linear = self.__init_linear__()
        if (initln): self.linear.apply(H._weights_init(mean=initln_mean, std=initln_std))
        if self.do_extlin:
            self.extlinear = nn.Linear(self.n_embd, self.n_embd)
            if (initln): self.extlinear.apply(H._weights_init(mean=initln_mean, std=initln_std))

    def __init_linear__(self):
        return nn.Sequential(nn.Linear(self.hdim, self.num_lbs), nn.Sigmoid()) if self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] else nn.Linear(self.hdim, self.num_lbs)

    def __lm_head__(self):
        return nn.Linear(self.lm_config.n_embd, self.lm_config.vocab_size, bias=False)


class TransformXLClfHead(BaseClfHead):
    def __init__(self, config, lm_model, lm_config, iactvtn='relu', oactvtn='sigmoid', fchdim=0, extfc=False, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, lm_loss=False, do_drop=True, pdrop=0.2, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, initln=False, initln_mean=0., initln_std=0.02, task_params={}, **kwargs):
        from util import config as C
        super(TransformXLClfHead, self).__init__(config, lm_model, lm_config, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=mlt_trnsfmr, lm_loss=lm_loss, do_drop=do_drop, pdrop=pdrop, do_norm=do_norm, do_lastdrop=do_lastdrop, do_crf=do_crf, task_params=task_params, **kwargs)
        self.vocab_size = lm_config.n_token
        self.n_embd = lm_config.d_embed
        self.norm = C.NORM_TYPE_MAP[norm_type](lm_config.n_embd)
        self._int_actvtn = C.ACTVTN_MAP[iactvtn]
        self._out_actvtn = C.ACTVTN_MAP[oactvtn]
        self.fchdim = fchdim
        self.extfc = extfc
        self.hdim = self.dim_mulriple * self.n_embd if self.mlt_trnsfmr and self.task_type in ['sentsim'] else self.n_embd
        self.linear = self.__init_linear__()
        if (initln): self.linear.apply(H._weights_init(mean=initln_mean, std=initln_std))

    def __init_linear__(self):
        return nn.Linear(self.hdim, self.num_lbs)

    def __lm_head__(self):
        pass

    def _lm_logit(self, input_ids, extra_inputs, hidden_states, all_hidden_states=None, extra_outputs={}):
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

    def _mlt_lm_logit(self, input_ids, extra_inputs, hidden_states, all_hidden_states=None, extra_outputs={}):
        return self._lm_logit(input_ids, hidden_states, extra_inputs, all_hidden_states=all_hidden_states, pool_idx=pool_idx)

    def transformer(self, input_ids, **extra_inputs):
        return self.lm_model.transformer(input_ids=input_ids.view(input_ids.size(0), -1))
