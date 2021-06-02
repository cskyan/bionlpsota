#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2021 by Caspar. All rights reserved.
# File Name: embedding.py
# Author: Shankai Yan
# E-mail: dr.skyan@gmail.com
# Created Time: 2021-03-29 17:19:30
###########################################################################
#

import os, copy, logging

import torch
from torch import nn

from allennlp.modules.conditional_random_field import ConditionalRandomField

from util import func as H
from . import transformer as T


class EmbeddingClfHead(T.BaseClfHead):
    def __init__(self, config, lm_model, lm_config, embed_type='w2v', w2v_path=None, iactvtn='relu', oactvtn='sigmoid', fchdim=0, extfc=False, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, lm_loss=False, do_drop=True, pdrop=0.2, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, do_thrshld=False, constraints=[], initln=False, initln_mean=0., initln_std=0.02, task_params={}, **kwargs):
        from util import config as C
        super(EmbeddingClfHead, self).__init__(config, lm_model, lm_config, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=config.task_type in ['entlmnt', 'sentsim'] and task_params.setdefault('sentsim_func', None) is not None, task_params=task_params, **kwargs)
        self.dim_mulriple = 2 if self.task_type in ['entlmnt', 'sentsim'] and (self.task_params.setdefault('sentsim_func', None) is None or self.task_params['sentsim_func'] == 'concat') else 1
        self.embed_type = embed_type
        if embed_type.startswith('w2v'):
            from gensim.models import KeyedVectors
            from gensim.models.keyedvectors import Word2VecKeyedVectors
            self.w2v_model = w2v_path if type(w2v_path) is Word2VecKeyedVectors else (KeyedVectors.load(w2v_path, mmap='r') if w2v_path and os.path.isfile(w2v_path) else None)
            assert(self.w2v_model)
            self.n_embd = self.w2v_model.syn0.shape[1] + (self.n_embd if hasattr(self, 'n_embd') else 0)
            config.register_callback('mdl_trsfm', EmbeddingClfHead.callback_update_w2v_model(self))
        elif embed_type.startswith('elmo'):
            self.vocab_size = 793471
            self.n_embd = lm_config['elmoedim'] * 2 + (self.n_embd if hasattr(self, 'n_embd') else 0) # two ELMo layer * ELMo embedding dimensions
            config.register_callback('mdl_trsfm', EmbeddingClfHead.callback_update_elmo_config(self))
        elif embed_type.startswith('elmo_w2v'):
            from gensim.models import KeyedVectors
            from gensim.models.keyedvectors import Word2VecKeyedVectors
            self.w2v_model = w2v_path if type(w2v_path) is Word2VecKeyedVectors else (KeyedVectors.load(w2v_path, mmap='r') if w2v_path and os.path.isfile(w2v_path) else None)
            assert(self.w2v_model)
            self.vocab_size = 793471
            self.n_embd = self.w2v_model.syn0.shape[1] + lm_config['elmoedim'] * 2 + (self.n_embd if hasattr(self, 'n_embd') else 0)
            config.register_callback('mdl_trsfm', EmbeddingClfHead.callback_update_w2v_model(self))
            config.register_callback('mdl_trsfm', EmbeddingClfHead.callback_update_elmo_config(self))
        self.norm = C.NORM_TYPE_MAP[norm_type](self.maxlen) if self.task_type == 'nmt' else C.NORM_TYPE_MAP[norm_type](self.n_embd)
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
        self.crf = ConditionalRandomField(num_lbs) if do_crf else None

    def __init_linear__(self):
        use_gpu = next(self.parameters()).is_cuda
        linear = (nn.Sequential(nn.Linear(self.hdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), *([] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.fchdim, self.num_lbs), self._out_actvtn()])) if self.task_type in ['entlmnt', 'sentsim'] else nn.Sequential(nn.Linear(self.hdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.num_lbs))) if self.fchdim else (nn.Sequential(*([nn.Linear(self.hdim, self.hdim), self._int_actvtn()] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.hdim, self.num_lbs), self._out_actvtn()])) if self.task_type in ['entlmnt', 'sentsim'] else nn.Linear(self.hdim, self.num_lbs))
        return linear.to('cuda') if use_gpu else linear

    def __lm_head__(self):
        return EmbeddingHead(self)

    def _w2v(self, input_ids, use_gpu=False):
        wembd_tnsr = torch.tensor([self.w2v_model.syn0[s] for s in input_ids])
        if use_gpu: wembd_tnsr = wembd_tnsr.to('cuda')
        return wembd_tnsr

    def _sentvec(self, input_ids, use_gpu=False):
        pass

    def forward(self, input_ids, *extra_inputs, labels=None, past=None, weights=None, embedding_mode=False, ret_mask=False):
        use_gpu = next(self.parameters()).is_cuda
        if self.sample_weights and len(extra_inputs) > 0:
            sample_weights = extra_inputs[-1]
            extra_inputs = extra_inputs[:-1]
        else:
            sample_weights = None
        unsolved_input_keys, unsolved_inputs = self.embed_type.split('_'), [input_ids]+list(extra_inputs)
        extra_inputs_dict = dict(zip([x for x in self.input_keys if x != 'input_ids'], extra_inputs))
        pool_idx = extra_inputs_dict['mask'].sum(1)
        mask = extra_inputs_dict['mask'] # mask of the original textual input
        clf_hs = []
        if self.task_type in ['entlmnt', 'sentsim']:
            if (self.embed_type.startswith('elmo')):
                embeddings = (self.lm_model(input_ids[0]), self.lm_model(input_ids[1]))
                clf_hs.append((torch.cat(embeddings[0]['elmo_representations'], dim=-1), torch.cat(embeddings[1]['elmo_representations'], dim=-1)))
                del unsolved_input_keys[0]
                del unsolved_inputs[0]
            for input_key, input_tnsr in zip(unsolved_input_keys, unsolved_inputs):
                clf_hs.append([getattr(self, '_%s'%input_key)(input_tnsr[x], use_gpu=use_gpu) for x in [0,1]])
            clf_h = [torch.cat(embds, dim=-1) for embds in zip(*clf_hs)]
        else:
            if (self.embed_type.startswith('elmo')):
                embeddings = self.lm_model(input_ids)
                clf_hs.append(torch.cat(embeddings['elmo_representations'], dim=-1))
                del unsolved_input_keys[0]
                del unsolved_inputs[0]
            for input_key, input_tnsr in zip(unsolved_input_keys, unsolved_inputs):
                clf_hs.append(getattr(self, '_%s'%input_key)(input_tnsr, use_gpu=use_gpu))
            clf_h = torch.cat(clf_hs, dim=-1)
        if labels is None:
            return (clf_h, mask) if ret_mask else (clf_h,)
        # Calculate language model loss
        if (self.lm_loss):
            lm_logits, lm_target = self.lm_logit(input_ids, clf_h, extra_inputs_dict)
            lm_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            lm_loss = lm_loss_func(lm_logits.contiguous().view(-1, lm_logits.size(-1)), lm_target.contiguous().view(-1)).view(input_ids.size(0), -1)
            if sample_weights is not None: lm_loss *= sample_weights
        else:
            lm_loss = None
        return (clf_h, lm_loss, mask) if ret_mask else (clf_h, lm_loss)

    def _forward(self, clf_h, mask, labels=None, weights=None): # For fine-tune task
        if self.task_type in ['entlmnt', 'sentsim']:
            if self.do_norm: clf_h = [self.norm(clf_h[x]) for x in [0,1]]
            clf_h = [self.dropout(clf_h[x]) for x in [0,1]]
            if (self.task_type == 'entlmnt' or self.task_params.setdefault('sentsim_func', None) is None or self.task_params['sentsim_func'] == 'concat'):
                if task_params.setdefault('concat_strategy', 'normal') == 'diff':
                    clf_h = torch.cat(clf_h+[torch.abs(clf_h[0]-clf_h[1]), clf_h[0]*clf_h[1]], dim=-1)
                elif task_params.setdefault('concat_strategy', 'normal') == 'flipflop':
                    clf_h = (torch.cat(clf_h, dim=-1) + torch.cat(clf_h[::-1], dim=-1))
                else:
                    clf_h = torch.cat(clf_h, dim=-1)
                clf_logits = self.linear(clf_h) if self.linear else clf_h
            else:
                clf_logits = clf_h = F.pairwise_distance(self.linear(clf_h[0]), self.linear(clf_h[1]), 2, eps=1e-12) if self.task_params['sentsim_func'] == 'dist' else F.cosine_similarity(self.linear(clf_h[0]), self.linear(clf_h[1]), dim=1, eps=1e-12)
        else:
            if self.do_norm: clf_h = self.norm(clf_h)
            clf_h = self.dropout(clf_h)
            clf_logits = self.linear(clf_h)
            if self.do_lastdrop: clf_logits = self.last_dropout(clf_logits)

        if (labels is None):
            if self.crf:
                tag_seq, score = zip(*self.crf.viterbi_tags(clf_logits.view(input_ids.size()[0], -1, self.num_lbs), torch.ones(*(input_ids.size()[:2])).int()))
                tag_seq = torch.tensor(tag_seq).to('cuda') if use_gpu else torch.tensor(tag_seq)
                clf_logits = torch.zeros((*tag_seq.size(), self.num_lbs)).to('cuda') if use_gpu else torch.zeros((*tag_seq.size(), self.num_lbs))
                clf_logits = clf_logits.scatter(-1, tag_seq.unsqueeze(-1), 1)
                return clf_logits
            if (self.task_type == 'sentsim' and self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != self.task_params['ymode']): return 1 - clf_logits.view(-1, self.num_lbs)
            return clf_logits.view(-1, self.num_lbs)
        if self.crf:
            clf_loss = -self.crf(clf_logits.view(input_ids.size()[0], -1, self.num_lbs), mask.long())
        elif self.task_type == 'mltc-clf' or self.task_type == 'entlmnt' or self.task_type == 'nmt':
            loss_func = nn.CrossEntropyLoss(weight=weights, reduction='none')
            clf_loss = loss_func(clf_logits.view(-1, self.num_lbs), labels.view(-1))
        elif self.task_type == 'mltl-clf':
            loss_func = nn.BCEWithLogitsLoss(pos_weight=10*weights if weights is not None else None, reduction='none')
            clf_loss = loss_func(clf_logits.view(-1, self.num_lbs), labels.view(-1, self.num_lbs).float())
        elif self.task_type == 'sentsim':
            from util import config as C
            loss_cls = C.RGRSN_LOSS_MAP[self.task_params.setdefault('loss', 'contrastive')]
            loss_func = loss_cls(reduction='none', x_mode=C.SIM_FUNC_MAP.setdefault(self.task_params['sentsim_func'], 'dist'), y_mode=self.task_params.setdefault('ymode', 'sim')) if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else nn.MSELoss(reduction='none')
            clf_loss = loss_func(clf_logits.view(-1), labels.view(-1))
        return clf_loss

    def _filter_vocab(self):
        pass

    @classmethod
    def callback_update_w2v_model(cls, model):
        def _callback(config):
            from util import config as C
            setattr(config, 'w2v_model', model.w2v_model)
            config.delayed_update(C.Configurable.PREDEFINED_MODEL_CONFIG_DELAYED_UPDATES[config.model])
        return _callback

    @classmethod
    def callback_update_elmo_config(cls, model):
        def _callback(config):
            from util import config as C
            setattr(config, 'lm_config', model.lm_config)
            config.delayed_update(C.Configurable.PREDEFINED_MODEL_CONFIG_DELAYED_UPDATES[config.model])
        return _callback

class EmbeddingPool(EmbeddingClfHead):
    def __init__(self, config, lm_model, lm_config, pooler=None, pool_params={'kernel_size':8, 'stride':4}, embed_type='w2v', w2v_path=None, iactvtn='relu', oactvtn='sigmoid', fchdim=0, extfc=False, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, lm_loss=False, do_drop=True, pdrop=0.2, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, do_thrshld=False, constraints=[], initln=False, initln_mean=0., initln_std=0.02, task_params={}, **kwargs):
        assert(config.task_type != 'nmt')
        from util import config as C
        super(EmbeddingPool, self).__init__(config, lm_model, lm_config, embed_type=embed_type, w2v_path=w2v_path, iactvtn=iactvtn, oactvtn=oactvtn, fchdim=fchdim, extfc=extfc, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=mlt_trnsfmr, lm_loss=lm_loss, do_drop=do_drop, pdrop=pdrop, do_norm=do_norm, norm_type=norm_type, do_lastdrop=do_lastdrop, do_crf=do_crf, do_thrshld=do_thrshld, constraints=constraints, initln=initln, initln_mean=initln_mean, initln_std=initln_std, task_params=task_params, **kwargs)
        self.maxlen = self.task_params.setdefault('maxlen', 128)
        if pooler:
            self.pooler = nn.MaxPool2d(**pool_params) if pooler == 'max' else nn.AvgPool2d(**pool_params)
            encoder_odim = int((2 * self.maxlen + 2 * pool_params.setdefault('padding', 0) - pool_params.setdefault('dilation', 1) * (pool_params['kernel_size'] - 1) - 1) / pool_params['stride'] + 1) * int((int(0.5 * self.n_embd) + 2 * pool_params.setdefault('padding', 0) - pool_params.setdefault('dilation', 1) * (pool_params['kernel_size'] - 1) - 1) / pool_params['stride'] + 1) if pooler == 'max' else int((2 * self.maxlen + 2 * pool_params.setdefault('padding', 0) - pool_params['kernel_size']) / pool_params['stride'] + 1) * int((int(0.5 * self.n_embd) + 2 * pool_params.setdefault('padding', 0) - pool_params['kernel_size']) / pool_params['stride'] + 1)
            self.norm = C.NORM_TYPE_MAP[norm_type](encoder_odim)
            self.hdim = self.dim_mulriple * encoder_odim if self.task_type in ['entlmnt', 'sentsim'] else encoder_odim
        else:
            self.pooler = None
            self.norm = C.NORM_TYPE_MAP[norm_type](self.n_embd)
            self.hdim = self.n_embd
        self.linear = self.__init_linear__()
        if (initln): self.linear.apply(H._weights_init(mean=initln_mean, std=initln_std))

    def forward(self, input_ids, *extra_inputs, labels=None, past=None, weights=None, embedding_mode=False):
        outputs = super(EmbeddingPool, self).forward(input_ids, *extra_inputs, labels=labels, past=past, weights=weights, embedding_mode=embedding_mode, ret_mask=True)
        if labels is None:
            clf_h, mask = outputs
        else:
            clf_h, lm_loss, mask = outputs
        pool_idx = mask.sum(1)
        if self.pooler:
            clf_h = [clf_h[x].view(clf_h[x].size(0), 2*clf_h[x].size(1), -1) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else clf_h.view(clf_h.size(0), 2*clf_h.size(1), -1)
            clf_h = [self.pooler(clf_h[x]).view(clf_h[x].size(0), -1) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else self.pooler(clf_h).view(clf_h.size(0), -1)
        else:
            clf_h = [clf_h[x].gather(1, pool_idx[x].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h[x].size(2))).squeeze(1) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else clf_h.gather(1, pool_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h.size(2))).squeeze(1)
        return (self._forward(clf_h, mask, labels=labels, weights=weights),) + (({},) if labels is None else (lm_loss, {}))


class EmbeddingSeq2Vec(EmbeddingClfHead):
    def __init__(self, config, lm_model, lm_config, seq2vec=None, s2v_params={'hdim':768}, embed_type='w2v', w2v_path=None, iactvtn='relu', oactvtn='sigmoid', fchdim=0, extfc=False, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, lm_loss=False, do_drop=True, pdrop=0.2, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, do_thrshld=False, constraints=[], initln=False, initln_mean=0., initln_std=0.02, task_params={}, **kwargs):
        assert(config.task_type != 'nmt')
        from util import config as C
        super(EmbeddingSeq2Vec, self).__init__(config, lm_model, lm_config, embed_type=embed_type, w2v_path=w2v_path, iactvtn=iactvtn, oactvtn=oactvtn, fchdim=fchdim, extfc=extfc, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=mlt_trnsfmr, lm_loss=lm_loss, do_drop=do_drop, pdrop=pdrop, do_norm=do_norm, norm_type=norm_type, do_lastdrop=do_lastdrop, do_crf=do_crf, do_thrshld=do_thrshld, constraints=constraints, initln=initln, initln_mean=initln_mean, initln_std=initln_std, task_params=task_params, **kwargs)
        if seq2vec:
            params = {}
            if seq2vec.startswith('pytorch-'):
                pth_mdl = '-'.join(seq2vec.split('-')[1:])
                _ = [params.update(x) for x in [C.SEQ2VEC_MDL_PARAMS.setdefault('pytorch', {}).setdefault(embed_type, {}), C.SEQ2VEC_TASK_PARAMS.setdefault('pytorch', {}).setdefault(self.task_type, {})]]
                _ = [params.update({p:s2v_params[k]}) for k, p in C.SEQ2VEC_LM_PARAMS_MAP.setdefault('pytorch', []) if k in s2v_params]
                if (embed_type == 'w2v'): params[pth_mdl]['input_size'] = self.w2v_model.syn0.shape[1]
                if (embed_type == 'elmo_w2v'): params[pth_mdl]['input_size'] = params[pth_mdl]['input_size'] + self.w2v_model.syn0.shape[1]
                self.seq2vec = H.gen_pytorch_wrapper('seq2vec', pth_mdl, **params[pth_mdl])
                encoder_odim = C.SEQ2VEC_DIM_INFER[seq2vec]([self.n_embd, self.dim_mulriple, params[pth_mdl]])
            else:
                _ = [params.update(x) for x in [C.SEQ2VEC_MDL_PARAMS.setdefault(seq2vec, {}).setdefault(embed_type, {}), C.SEQ2VEC_TASK_PARAMS.setdefault(seq2vec, {}).setdefault(self.task_type, {})]]
                _ = [params.update({p:s2v_params[k]}) for k, p in C.SEQ2VEC_LM_PARAMS_MAP.setdefault(seq2vec, []) if k in s2v_params]
                if (embed_type == 'w2v'): params['embedding_dim'] = self.w2v_model.syn0.shape[1]
                if (embed_type == 'elmo_w2v'): params['embedding_dim'] = params['embedding_dim'] + self.w2v_model.syn0.shape[1]
                self.seq2vec = C.SEQ2VEC_MAP[seq2vec](**params)
                if hasattr(self.seq2vec, 'get_output_dim') and seq2vec != 'boe':
                    encoder_odim = self.seq2vec.get_output_dim()
                else:
                    encoder_odim = C.SEQ2VEC_DIM_INFER[seq2vec]([self.n_embd, self.dim_mulriple, params])
        else:
            self.seq2vec = None
            encoder_odim = self.n_embd
        self.maxlen = self.task_params.setdefault('maxlen', 128)
        self.norm = C.NORM_TYPE_MAP[norm_type](encoder_odim)
        self.hdim = self.dim_mulriple * encoder_odim if self.task_type in ['entlmnt', 'sentsim'] else encoder_odim
        self.linear = self.__init_linear__()
        if (self.linear and initln): self.linear.apply(H._weights_init(mean=initln_mean, std=initln_std))

    def forward(self, input_ids, *extra_inputs, labels=None, past=None, weights=None, embedding_mode=False):
        outputs = super(EmbeddingSeq2Vec, self).forward(input_ids, *extra_inputs, labels=labels, past=past, weights=weights, embedding_mode=embedding_mode, ret_mask=True)
        if labels is None:
            clf_h, mask = outputs
        else:
            clf_h, lm_loss, mask = outputs
        pool_idx = mask.sum(1)
        if self.seq2vec:
            clf_h = [self.seq2vec(clf_h[x], mask=mask[x]) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else self.seq2vec(clf_h, mask=mask)
        else:
            clf_h = [clf_h[x].gather(1, pool_idx[x].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h[x].size(2))).squeeze(1) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else clf_h.gather(1, pool_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h.size(2))).squeeze(1)
        return (self._forward(clf_h, mask, labels=labels, weights=weights),) + (({},) if labels is None else (lm_loss, {}))


class EmbeddingSeq2Seq(EmbeddingClfHead):
    def __init__(self, config, lm_model, lm_config, seq2seq=None, s2s_params={'hdim':768}, embed_type='w2v', w2v_path=None, iactvtn='relu', oactvtn='sigmoid', fchdim=0, extfc=False, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, lm_loss=False, do_drop=True, pdrop=0.2, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, do_thrshld=False, constraints=[], initln=False, initln_mean=0., initln_std=0.02, task_params={}, **kwargs):
        from util import config as C
        super(EmbeddingSeq2Seq, self).__init__(config, lm_model, lm_config, embed_type=embed_type, w2v_path=w2v_path, iactvtn=iactvtn, oactvtn=oactvtn, fchdim=fchdim, extfc=extfc, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=mlt_trnsfmr, lm_loss=lm_loss, do_drop=do_drop, pdrop=pdrop, do_norm=do_norm, norm_type=norm_type, do_lastdrop=do_lastdrop, do_crf=do_crf, do_thrshld=do_thrshld, constraints=constraints, initln=initln, initln_mean=initln_mean, initln_std=initln_std, task_params=task_params, **kwargs)
        if seq2seq:
            params = {}
            if seq2seq.startswith('pytorch-'):
                pth_mdl = '-'.join(seq2seq.split('-')[1:])
                _ = [params.update(x) for x in [C.SEQ2SEQ_MDL_PARAMS.setdefault('pytorch', {}).setdefault('elmo', {}), C.SEQ2SEQ_TASK_PARAMS.setdefault(seq2seq, {}).setdefault(self.task_type, {})]]
                self.seq2seq = H.gen_pytorch_wrapper('seq2seq', pth_mdl, **params[pth_mdl])
                encoder_odim = C.SEQ2SEQ_DIM_INFER[seq2seq]([self.n_embd, self.dim_mulriple, params[pth_mdl]])
            else:
                _ = [params.update(x) for x in [C.SEQ2SEQ_MDL_PARAMS.setdefault(seq2seq, {}).setdefault('elmo', {}), C.SEQ2SEQ_TASK_PARAMS.setdefault(seq2seq, {}).setdefault(self.task_type, {})]]
                self.seq2seq = C.SEQ2SEQ_MAP[seq2seq](**params)
                if hasattr(self.seq2seq, 'get_output_dim'):
                    encoder_odim = self.seq2seq.get_output_dim()
                else:
                    encoder_odim = C.SEQ2SEQ_DIM_INFER[seq2seq]([self.n_embd, self.dim_mulriple, params])
        else:
            self.seq2seq = None
            encoder_odim = self.n_embd
        self.maxlen = self.task_params.setdefault('maxlen', 128)
        self.norm = C.NORM_TYPE_MAP[norm_type](self.maxlen)
        # self.norm = nn.LayerNorm([128,2048])
        self.hdim = encoder_odim
        self.linear = self.__init_linear__()
        if (initln): self.linear.apply(H._weights_init(mean=initln_mean, std=initln_std))

    def __init_linear__(self):
        use_gpu = next(self.parameters()).is_cuda
        linear = nn.Sequential(nn.Linear(self.hdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.num_lbs), self._out_actvtn()) if self.fchdim else nn.Sequential(nn.Linear(self.hdim, self.num_lbs), self._out_actvtn())
        return linear.to('cuda') if use_gpu else linear

    def forward(self, input_ids, *extra_inputs, labels=None, past=None, weights=None, embedding_mode=False):
        clf_h, lm_loss, mask = super(EmbeddingSeq2Seq, self).forward(input_ids, *extra_inputs, labels=labels, past=past, weights=weights, embedding_mode=embedding_mode, ret_mask=True)
        if labels is None:
            clf_h, mask = outputs
        else:
            clf_h, lm_loss, mask = outputs
        if self.seq2seq:
            clf_h = self.seq2seq(clf_h, mask=mask)
        return (self._forward(clf_h, mask, labels=labels, weights=weights),) + (({},) if labels is None else (lm_loss, {}))


class SentVecEmbeddingSeq2Vec(EmbeddingSeq2Vec):
    def __init__(self, config, lm_model, lm_config, sentvec_path=None, seq2vec=None, s2v_params={'hdim':768}, embed_type='w2v_sentvec', w2v_path=None, iactvtn='relu', oactvtn='sigmoid', fchdim=0, extfc=False, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, lm_loss=False, do_drop=True, pdrop=0.2, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, do_thrshld=False, constraints=[], initln=False, initln_mean=0., initln_std=0.02, task_params={}, **kwargs):
        import sent2vec
        if type(sentvec_path) is sent2vec.Sent2vecModel:
            self.sentvec_model = w2v_path
        elif sentvec_path and os.path.isfile(sentvec_path):
            self.sentvec_model = sent2vec.Sent2vecModel()
            self.sentvec_model.load_model(sentvec_path)
        else:
            self.sentvec_model = None
        assert(self.sentvec_model)
        self.n_embd = self.sentvec_model.get_emb_size()
        super(SentVecEmbeddingSeq2Vec, self).__init__(config, lm_model, lm_config, seq2vec=seq2vec, s2v_params=s2v_params, embed_type=embed_type.replace('_sentvec', ''), w2v_path=w2v_path, iactvtn=iactvtn, oactvtn=oactvtn, fchdim=fchdim, extfc=extfc, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=mlt_trnsfmr, lm_loss=lm_loss, do_drop=do_drop, pdrop=pdrop, do_norm=do_norm, norm_type=norm_type, do_lastdrop=do_lastdrop, do_crf=do_crf, do_thrshld=do_thrshld, constraints=constraints, initln=initln, initln_mean=initln_mean, initln_std=initln_std, task_params=task_params, **kwargs)

    def forward(self, input_ids, *extra_inputs, labels=None, past=None, weights=None, embedding_mode=False):
        sample_weights, entvec_tnsr, extra_inputs = (extra_inputs[0], extra_inputs[1], extra_inputs[2:]) if self.sample_weights else (None, extra_inputs[0], extra_inputs[1:])
        outputs = EmbeddingClfHead.forward(self, input_ids, *extra_inputs, labels=labels, past=past, weights=weights, embedding_mode=embedding_mode, ret_mask=True)
        if labels is None:
            clf_h, mask = outputs
        else:
            clf_h, lm_loss, mask = outputs
        pool_idx = mask.sum(1)
        if self.seq2vec:
            clf_h = [self.seq2vec(clf_h[x], mask=mask[x]) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else self.seq2vec(clf_h, mask=mask)
        else:
            clf_h = [clf_h[x].gather(1, pool_idx[x].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h[x].size(2))).squeeze(1) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else clf_h.gather(1, pool_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h.size(2))).squeeze(1)
        clf_h = [torch.cat([clf_h[x], sentvec_tnsr[x]], dim=-1) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else torch.cat([clf_h, sentvec_tnsr], dim=-1)
        return (self._forward(clf_h, mask, labels=labels, weights=weights),) + (({},) if labels is None else (lm_loss, {}))


class EmbeddingHead(nn.Module):
    def __init__(self, base_model):
        super(EmbeddingHead, self).__init__()
        self.base_model = dict(zip(['model'], [base_model]))

    def forward(self, hidden_states, mask, labels=None): # For language model task
        use_gpu = next(self.base_model['model'].parameters()).is_cuda
        clf_h = hidden_states
        pool_idx = mask.sum(1)
        if (self.base_model['model'].task_params.setdefault('sentsim_func', None) == 'concat'):
            if self.base_model['model'].task_params.setdefault('concat_strategy', 'normal') == 'diff':
                clf_h = torch.cat(clf_h+[torch.abs(clf_h[0]-clf_h[1]), clf_h[0]*clf_h[1]], dim=-1)
            elif self.base_model['model'].task_params.setdefault('concat_strategy', 'normal') == 'flipflop':
                clf_h = (torch.cat(clf_h, dim=-1) + torch.cat(clf_h[::-1], dim=-1))
            else:
                clf_h = torch.cat(clf_h, dim=-1)
            clf_logits = self.base_model['model'].linear(clf_h) if self.base_model['model'].linear else clf_h
        else:
            clf_logits = clf_h = F.pairwise_distance(self.base_model['model'].linear(clf_h[0]), self.base_model['model'].linear(clf_h[1]), 2, eps=1e-12) if self.base_model['model'].task_params['sentsim_func'] == 'dist' else F.cosine_similarity(self.base_model['model'].linear(clf_h[0]), self.base_model['model'].linear(clf_h[1]), dim=1, eps=1e-12)
        if self.base_model['model'].thrshlder: self.base_model['model'].thrshld = self.base_model['model'].thrshlder(clf_h)
        if self.base_model['model'].do_lastdrop: clf_logits = self.last_dropout(clf_logits)

        if (labels is None):
            if self.base_model['model'].crf:
                tag_seq, score = zip(*self.base_model['model'].crf.viterbi_tags(clf_logits.view(input_ids.size()[0], -1, self.base_model['model'].num_lbs), torch.ones_like(input_ids)))
                tag_seq = torch.tensor(tag_seq).to('cuda') if use_gpu else torch.tensor(tag_seq)
                logging.debug((tag_seq.min(), tag_seq.max(), score))
                clf_logits = torch.zeros((*tag_seq.size(), self.base_model['model'].num_lbs)).to('cuda') if use_gpu else torch.zeros((*tag_seq.size(), self.base_model['model'].num_lbs))
                clf_logits = clf_logits.scatter(-1, tag_seq.unsqueeze(-1), 1)
                return clf_logits
            for cnstrnt in self.base_model['model'].constraints: clf_logits = cnstrnt(clf_logits)
            if (self.base_model['model'].mlt_trnsfmr and self.base_model['model'].task_type in ['entlmnt', 'sentsim'] and self.base_model['model'].task_params.setdefault('sentsim_func', None) is not None and self.base_model['model'].task_params['sentsim_func'] != 'concat' and self.base_model['model'].task_params['sentsim_func'] != self.base_model['model'].task_params.setdefault('ymode', 'sim')): return 1 - clf_logits.view(-1, self.base_model['model'].num_lbs)
            return clf_logits.view(-1, self.base_model['model'].num_lbs)
        if self.base_model['model'].crf:
            clf_loss = -self.base_model['model'].crf(clf_logits.view(input_ids.size()[0], -1, self.base_model['model'].num_lbs), pool_idx)
            if sample_weights is not None: clf_loss *= sample_weights
            return clf_loss, None
        else:
            for cnstrnt in self.base_model['model'].constraints: clf_logits = cnstrnt(clf_logits)
        if self.base_model['model'].task_type == 'mltc-clf' or (self.base_model['model'].task_type == 'entlmnt' and self.base_model['model'].num_lbs > 1) or self.base_model['model'].task_type == 'nmt':
            loss_func = nn.CrossEntropyLoss(weight=weights, reduction='none')
            clf_loss = loss_func(clf_logits.view(-1, self.base_model['model'].num_lbs), labels.view(-1))
        elif self.base_model['model'].task_type == 'mltl-clf' or (self.base_model['model'].task_type == 'entlmnt' and self.base_model['model'].num_lbs == 1):
            loss_func = nn.BCEWithLogitsLoss(pos_weight=10*weights if weights is not None else None, reduction='none')
            clf_loss = loss_func(clf_logits.view(-1, self.base_model['model'].num_lbs), labels.view(-1, self.base_model['model'].num_lbs).float())
        elif self.base_model['model'].task_type == 'sentsim':
            from util import config as C
            loss_cls = C.RGRSN_LOSS_MAP[self.base_model['model'].task_params.setdefault('loss', 'contrastive' if self.base_model['model'].task_params.setdefault('sentsim_func', None) and self.base_model['model'].task_params['sentsim_func'] != 'concat' else 'mse')]
            loss_func = loss_cls(reduction='none', x_mode=C.SIM_FUNC_MAP.setdefault(self.base_model['model'].task_params['sentsim_func'], 'dist'), y_mode=self.base_model['model'].task_params.setdefault('ymode', 'sim')) if self.base_model['model'].task_params.setdefault('sentsim_func', None) and self.base_model['model'].task_params['sentsim_func'] != 'concat' else (loss_cls(reduction='none', x_mode='sim', y_mode=self.base_model['model'].task_params.setdefault('ymode', 'sim')) if self.base_model['model'].task_params['sentsim_func'] == 'concat' else nn.MSELoss(reduction='none'))
            clf_loss = loss_func(clf_logits.view(-1), labels.view(-1))
        if self.base_model['model'].thrshlder:
            num_lbs = labels.view(-1, self.base_model['model'].num_lbs).sum(1)
            clf_loss = 0.8 * clf_loss + 0.2 * F.mse_loss(self.base_model['model'].thrshld, torch.sigmoid(torch.topk(clf_logits, k=num_lbs.max(), dim=1, sorted=True)[0][:,num_lbs-1]), reduction='mean')
        if sample_weights is not None: clf_loss *= sample_weights
        return clf_loss, lm_loss
