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

import torch
from torch import nn

from allennlp.modules.conditional_random_field import ConditionalRandomField

from util.func import gen_pytorch_wrapper

from . import transformer as T


class EmbeddingClfHead(T.BaseClfHead):
    def __init__(self, lm_model, config, task_type, iactvtn='relu', oactvtn='sigmoid', fchdim=0, embed_type='w2v', w2v_path=None, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, pdrop=0.2, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, initln=False, initln_mean=0., initln_std=0.02, task_params={}, **kwargs):
        from util import config as C
        super(EmbeddingClfHead, self).__init__(lm_model, config, task_type, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=task_type in ['entlmnt', 'sentsim'], pdrop=pdrop, do_norm=do_norm, do_lastdrop=do_lastdrop, do_crf=do_crf, task_params=task_params, **kwargs)
        self.dim_mulriple = 2 if self.task_type in ['entlmnt', 'sentsim'] and (self.task_params.setdefault('sentsim_func', None) is None or self.task_params['sentsim_func'] == 'concat') else 1
        self.embed_type = embed_type
        if embed_type.startswith('w2v'):
            from gensim.models import KeyedVectors
            from gensim.models.keyedvectors import Word2VecKeyedVectors
            self.w2v_model = w2v_path if type(w2v_path) is Word2VecKeyedVectors else (KeyedVectors.load(w2v_path, mmap='r') if w2v_path and os.path.isfile(w2v_path) else None)
            assert(self.w2v_model)
            self.n_embd = self.w2v_model.syn0.shape[1] + (self.n_embd if hasattr(self, 'n_embd') else 0)
        elif embed_type.startswith('elmo'):
            self.vocab_size = 793471
            self.n_embd = config['elmoedim'] * 2 + (self.n_embd if hasattr(self, 'n_embd') else 0) # two ELMo layer * ELMo embedding dimensions
        elif embed_type.startswith('elmo_w2v'):
            from gensim.models import KeyedVectors
            from gensim.models.keyedvectors import Word2VecKeyedVectors
            self.w2v_model = w2v_path if type(w2v_path) is Word2VecKeyedVectors else (KeyedVectors.load(w2v_path, mmap='r') if w2v_path and os.path.isfile(w2v_path) else None)
            assert(self.w2v_model)
            self.vocab_size = 793471
            self.n_embd = self.w2v_model.syn0.shape[1] + config['elmoedim'] * 2 + (self.n_embd if hasattr(self, 'n_embd') else 0)
        self._int_actvtn = C.ACTVTN_MAP[iactvtn]
        self._out_actvtn = C.ACTVTN_MAP[oactvtn]
        self.fchdim = fchdim
        self.crf = ConditionalRandomField(num_lbs) if do_crf else None

    def forward(self, input_ids, pool_idx, w2v_ids=None, labels=None, past=None, weights=None, ret_mask=False):
        use_gpu = next(self.parameters()).is_cuda
        if self.task_type in ['entlmnt', 'sentsim']:
            mask = [torch.arange(input_ids[x].size()[1]).to('cuda').unsqueeze(0).expand(input_ids[x].size()[:2]) <= pool_idx[x].unsqueeze(1).expand(input_ids[x].size()[:2]) if use_gpu else torch.arange(input_ids[x].size()[1]).unsqueeze(0).expand(input_ids[x].size()[:2]) <= pool_idx[x].unsqueeze(1).expand(input_ids[x].size()[:2]) for x in [0,1]]
            if (self.embed_type.startswith('w2v')):
                assert(w2v_ids is not None and self.w2v_model is not None)
                wembd_tnsr = [torch.tensor([self.w2v_model.syn0[s] for s in w2v_ids[x]]) for x in [0,1]]
                if use_gpu: wembd_tnsr = [x.to('cuda') for x in wembd_tnsr]
                clf_h = wembd_tnsr
            elif (self.embed_type.startswith('elmo')):
                embeddings = (self.lm_model(input_ids[0]), self.lm_model(input_ids[1]))
                clf_h = torch.cat(embeddings[0]['elmo_representations'], dim=-1), torch.cat(embeddings[1]['elmo_representations'], dim=-1)
            elif (self.embed_type.startswith('elmo_w2v')):
                assert(w2v_ids is not None and self.w2v_model is not None)
                wembd_tnsr = [torch.tensor([self.w2v_model.syn0[s] for s in w2v_ids[x]]) for x in [0,1]]
                if use_gpu: wembd_tnsr = [x.to('cuda') for x in wembd_tnsr]
                elmo_embeddings = (self.lm_model(input_ids[0]), self.lm_model(input_ids[1]))
                elmo_clf_h = torch.cat(elmo_embeddings[0]['elmo_representations'], dim=-1), torch.cat(elmo_embeddings[1]['elmo_representations'], dim=-1)
                clf_h = [torch.cat([elmoembd, wembd], dim=-1) for elmoembd, wembd in zip(elmo_clf_h, wembd_tnsr)]
        else:
            mask = torch.arange(input_ids.size()[1]).to('cuda').unsqueeze(0).expand(input_ids.size()[:2]) <= pool_idx.unsqueeze(1).expand(input_ids.size()[:2]) if use_gpu else torch.arange(input_ids.size()[1]).unsqueeze(0).expand(input_ids.size()[:2]) <= pool_idx.unsqueeze(1).expand(input_ids.size()[:2])
            if (self.embed_type.startswith('w2v')):
                assert(w2v_ids is not None)
                wembd_tnsr = torch.tensor([self.w2v_model.syn0[s] for s in w2v_ids])
                if use_gpu: wembd_tnsr = wembd_tnsr.to('cuda')
                clf_h = wembd_tnsr
            elif (self.embed_type.startswith('elmo')):
                embeddings = self.lm_model(input_ids)
                clf_h = torch.cat(embeddings['elmo_representations'], dim=-1)
            elif (self.embed_type.startswith('elmo_w2v')):
                assert(w2v_ids is not None)
                wembd_tnsr = torch.tensor([self.w2v_model.syn0[s] for s in w2v_ids])
                if use_gpu: wembd_tnsr = wembd_tnsr.to('cuda')
                elmo_embeddings = self.lm_model(input_ids)
                elmo_clf_h = torch.cat(elmo_embeddings['elmo_representations'], dim=-1)
                clf_h = torch.cat([elmo_clf_h, wembd_tnsr], dim=-1)
        return (clf_h, mask) if ret_mask else clf_h

    def _forward(self, clf_h, labels=None, weights=None):
        if self.task_type in ['entlmnt', 'sentsim']:
            if self.do_norm: clf_h = [self.norm(clf_h[x]) for x in [0,1]]
            clf_h = [self.dropout(clf_h[x]) for x in [0,1]]
            if (self.task_type == 'entlmnt' or self.task_params.setdefault('sentsim_func', None) is None or self.task_params['sentsim_func'] == 'concat'):
                # clf_h = (torch.cat(clf_h, dim=-1) + torch.cat(clf_h[::-1], dim=-1))
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
        return clf_loss, None

    def _filter_vocab(self):
        pass


class EmbeddingPool(EmbeddingClfHead):
    def __init__(self, lm_model, config, task_type, iactvtn='relu', oactvtn='sigmoid', fchdim=0, embed_type='w2v', w2v_path=None, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, pdrop=0.2, pooler=None, pool_params={'kernel_size':8, 'stride':4}, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, initln=False, initln_mean=0., initln_std=0.02, task_params={}, **kwargs):
        assert(task_type != 'nmt')
        from util import config as C
        super(EmbeddingPool, self).__init__(lm_model, config, task_type, iactvtn=iactvtn, oactvtn=oactvtn, fchdim=fchdim, embed_type=embed_type, w2v_path=w2v_path, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=mlt_trnsfmr, pdrop=pdrop, do_norm=do_norm, norm_type=norm_type, do_lastdrop=do_lastdrop, do_crf=do_crf, initln=initln, initln_mean=initln_mean, initln_std=initln_std, task_params=task_params, **kwargs)
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
        if (initln): self.linear.apply(_weights_init(mean=initln_mean, std=initln_std))

    def __init_linear__(self):
        use_gpu = next(self.parameters()).is_cuda
        linear = (nn.Sequential(nn.Linear(self.hdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), *([] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.fchdim, self.num_lbs), self._out_actvtn()])) if self.task_type in ['entlmnt', 'sentsim'] else nn.Sequential(nn.Linear(self.hdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.num_lbs))) if self.fchdim else (nn.Sequential(*([nn.Linear(self.hdim, self.hdim), self._int_actvtn()] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.hdim, self.num_lbs), self._out_actvtn()])) if self.task_type in ['entlmnt', 'sentsim'] else nn.Linear(self.hdim, self.num_lbs))
        return linear.to('cuda') if use_gpu else linear

    def forward(self, input_ids, pool_idx, w2v_ids=None, labels=None, past=None, weights=None):
        clf_h = super(EmbeddingPool, self).forward(input_ids, pool_idx, w2v_ids=w2v_ids, labels=labels, past=past, weights=weights)
        if self.pooler:
            clf_h = [clf_h[x].view(clf_h[x].size(0), 2*clf_h[x].size(1), -1) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else clf_h.view(clf_h.size(0), 2*clf_h.size(1), -1)
            clf_h = [self.pooler(clf_h[x]).view(clf_h[x].size(0), -1) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else self.pooler(clf_h).view(clf_h.size(0), -1)
        else:
            clf_h = [clf_h[x].gather(1, pool_idx[x].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h[x].size(2))).squeeze(1) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else clf_h.gather(1, pool_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h.size(2))).squeeze(1)
        return self._forward(clf_h, labels=labels, weights=weights)


class EmbeddingSeq2Vec(EmbeddingClfHead):
    def __init__(self, lm_model, config, task_type, iactvtn='relu', oactvtn='sigmoid', fchdim=0, embed_type='w2v', w2v_path=None, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, pdrop=0.2, seq2vec=None, s2v_params={'hdim':768}, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, initln=False, initln_mean=0., initln_std=0.02, task_params={}, **kwargs):
        assert(task_type != 'nmt')
        from util import config as C
        super(EmbeddingSeq2Vec, self).__init__(lm_model, config, task_type, iactvtn=iactvtn, oactvtn=oactvtn, fchdim=fchdim, embed_type=embed_type, w2v_path=w2v_path, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=mlt_trnsfmr, pdrop=pdrop, do_norm=do_norm, norm_type=norm_type, do_lastdrop=do_lastdrop, do_crf=do_crf, initln=initln, initln_mean=initln_mean, initln_std=initln_std, task_params=task_params, **kwargs)
        if seq2vec:
            params = {}
            if seq2vec.startswith('pytorch-'):
                pth_mdl = '-'.join(seq2vec.split('-')[1:])
                _ = [params.update(x) for x in [C.SEQ2VEC_MDL_PARAMS.setdefault('pytorch', {}).setdefault(embed_type, {}), C.SEQ2VEC_TASK_PARAMS.setdefault('pytorch', {}).setdefault(task_type, {})]]
                _ = [params.update({p:s2v_params[k]}) for k, p in C.SEQ2VEC_LM_PARAMS_MAP.setdefault('pytorch', []) if k in s2v_params]
                if (embed_type == 'w2v'): params[pth_mdl]['input_size'] = self.w2v_model.syn0.shape[1]
                if (embed_type == 'elmo_w2v'): params[pth_mdl]['input_size'] = params[pth_mdl]['input_size'] + self.w2v_model.syn0.shape[1]
                self.seq2vec = gen_pytorch_wrapper('seq2vec', pth_mdl, **params[pth_mdl])
                encoder_odim = C.SEQ2VEC_DIM_INFER[seq2vec]([self.n_embd, self.dim_mulriple, params[pth_mdl]])
            else:
                _ = [params.update(x) for x in [C.SEQ2VEC_MDL_PARAMS.setdefault(seq2vec, {}).setdefault(embed_type, {}), C.SEQ2VEC_TASK_PARAMS.setdefault(seq2vec, {}).setdefault(task_type, {})]]
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
        if (self.linear and initln): self.linear.apply(_weights_init(mean=initln_mean, std=initln_std))

    def __init_linear__(self):
        use_gpu = next(self.parameters()).is_cuda
        linear = (nn.Sequential(nn.Linear(self.hdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), *([] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.fchdim, self.num_lbs), self._out_actvtn()])) if self.task_type in ['entlmnt', 'sentsim'] else nn.Sequential(nn.Linear(self.hdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.num_lbs))) if self.fchdim else (nn.Sequential(*([nn.Linear(self.hdim, self.hdim), self._int_actvtn()] if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else [nn.Linear(self.hdim, self.num_lbs), self._out_actvtn()])) if self.task_type in ['entlmnt', 'sentsim'] else nn.Linear(self.hdim, self.num_lbs))
        return linear.to('cuda') if use_gpu else linear

    def forward(self, input_ids, pool_idx, w2v_ids=None, labels=None, past=None, weights=None):
        clf_h, mask = super(EmbeddingSeq2Vec, self).forward(input_ids, pool_idx, w2v_ids=w2v_ids, labels=labels, past=past, weights=weights, ret_mask=True)
        if self.seq2vec:
            clf_h = [self.seq2vec(clf_h[x], mask=mask[x]) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else self.seq2vec(clf_h, mask=mask)
        else:
            clf_h = [clf_h[x].gather(1, pool_idx[x].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h[x].size(2))).squeeze(1) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else clf_h.gather(1, pool_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h.size(2))).squeeze(1)
        return self._forward(clf_h, labels=labels, weights=weights)


class EmbeddingSeq2Seq(EmbeddingClfHead):
    def __init__(self, lm_model, config, task_type, iactvtn='relu', oactvtn='sigmoid', fchdim=0, embed_type='w2v', w2v_path=None, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, pdrop=0.2, seq2seq=None, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, initln=False, initln_mean=0., initln_std=0.02, task_params={}, **kwargs):
        from util import config as C
        super(EmbeddingSeq2Seq, self).__init__(lm_model, config, task_type, iactvtn=iactvtn, oactvtn=oactvtn, fchdim=fchdim, embed_type=embed_type, w2v_path=w2v_path, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=mlt_trnsfmr, pdrop=pdrop, do_norm=do_norm, norm_type=norm_type, do_lastdrop=do_lastdrop, do_crf=do_crf, initln=initln, initln_mean=initln_mean, initln_std=initln_std, task_params=task_params, **kwargs)
        if seq2seq:
            params = {}
            if seq2seq.startswith('pytorch-'):
                pth_mdl = '-'.join(seq2seq.split('-')[1:])
                _ = [params.update(x) for x in [C.SEQ2SEQ_MDL_PARAMS.setdefault('pytorch', {}).setdefault('elmo', {}), C.SEQ2SEQ_TASK_PARAMS.setdefault(seq2seq, {}).setdefault(task_type, {})]]
                self.seq2seq = gen_pytorch_wrapper('seq2seq', pth_mdl, **params[pth_mdl])
                encoder_odim = C.SEQ2SEQ_DIM_INFER[seq2seq]([self.n_embd, self.dim_mulriple, params[pth_mdl]])
            else:
                _ = [params.update(x) for x in [C.SEQ2SEQ_MDL_PARAMS.setdefault(seq2seq, {}).setdefault('elmo', {}), C.SEQ2SEQ_TASK_PARAMS.setdefault(seq2seq, {}).setdefault(task_type, {})]]
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
        if (initln): self.linear.apply(_weights_init(mean=initln_mean, std=initln_std))

    def __init_linear__(self):
        use_gpu = next(self.parameters()).is_cuda
        linear = nn.Sequential(nn.Linear(self.hdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.fchdim), self._int_actvtn(), nn.Linear(self.fchdim, self.num_lbs), self._out_actvtn()) if self.fchdim else nn.Sequential(nn.Linear(self.hdim, self.num_lbs), self._out_actvtn())
        return linear.to('cuda') if use_gpu else linear

    def forward(self, input_ids, pool_idx, w2v_ids=None, labels=None, past=None, weights=None):
        clf_h, mask = super(EmbeddingSeq2Seq, self).forward(input_ids, pool_idx, w2v_ids=w2v_ids, labels=labels, past=past, weights=weights, ret_mask=True)
        if self.seq2seq:
            clf_h = self.seq2seq(clf_h, mask=mask)
        return self._forward(clf_h, labels=labels, weights=weights)


class SentVecEmbeddingSeq2Vec(EmbeddingSeq2Vec):
    def __init__(self, lm_model, config, task_type, iactvtn='relu', oactvtn='sigmoid', fchdim=0, embed_type='w2v_sentvec', w2v_path=None, sentvec_path=None, sample_weights=False, num_lbs=1, mlt_trnsfmr=False, pdrop=0.2, seq2vec=None, s2v_params={'hdim':768}, do_norm=True, norm_type='batch', do_lastdrop=True, do_crf=False, initln=False, initln_mean=0., initln_std=0.02, task_params={}, **kwargs):
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
        super(SentVecEmbeddingSeq2Vec, self).__init__(lm_model, config, task_type, iactvtn=iactvtn, oactvtn=oactvtn, fchdim=fchdim, embed_type=embed_type.replace('_sentvec', ''), w2v_path=w2v_path, sample_weights=sample_weights, num_lbs=num_lbs, mlt_trnsfmr=mlt_trnsfmr, pdrop=pdrop, do_norm=do_norm, norm_type=norm_type, do_lastdrop=do_lastdrop, do_crf=do_crf, initln=initln, initln_mean=initln_mean, initln_std=initln_std, task_params=task_params, **kwargs)

    def forward(self, input_ids, pool_idx, *extra_inputs, w2v_ids=None, labels=None, past=None, weights=None):
        sample_weights, entvec_tnsr, extra_inputs = (extra_inputs[0], extra_inputs[1], extra_inputs[2:]) if self.sample_weights else (None, extra_inputs[0], extra_inputs[1:])
        clf_h, mask = EmbeddingClfHead.forward(self, input_ids, pool_idx, *extra_inputs, w2v_ids=w2v_ids, labels=labels, past=past, weights=weights, ret_mask=True)
        if self.seq2vec:
            clf_h = [self.seq2vec(clf_h[x], mask=mask[x]) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else self.seq2vec(clf_h, mask=mask)
        else:
            clf_h = [clf_h[x].gather(1, pool_idx[x].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h[x].size(2))).squeeze(1) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else clf_h.gather(1, pool_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h.size(2))).squeeze(1)
        clf_h = [torch.cat([clf_h[x], sentvec_tnsr[x]], dim=-1) for x in [0,1]] if self.task_type in ['entlmnt', 'sentsim'] else torch.cat([clf_h, sentvec_tnsr], dim=-1)
        return self._forward(clf_h, labels=labels, weights=weights)


class EmbeddingHead(nn.Module):
    def __init__(self, base_model):
        super(EmbeddingHead, self).__init__()
        self.base_model = dict(zip(['model'], [base_model]))
        self.linear = self.base_model['model'].linear
        self.task_type = self.base_model['model'].task_type
        self.task_params = self.base_model['model'].task_params
        self.num_lbs = self.base_model['model'].num_lbs
        self.mlt_trnsfmr = self.base_model['model'].mlt_trnsfmr
        self.thrshld = self.base_model['model'].thrshld
        self.thrshlder = self.base_model['model'].thrshlder
        self.do_lastdrop = self.base_model['model'].do_lastdrop
        self.crf = self.base_model['model'].crf
        self.constraints = self.base_model['model'].constraints

    def forward(self, hidden_states, labels=None):
        use_gpu = next(self.base_model['model'].parameters()).is_cuda
        clf_h = hidden_states
        if (self.task_params.setdefault('sentsim_func', None) == 'concat'):
            clf_h = torch.cat(clf_h+[torch.abs(clf_h[0]-clf_h[1]), clf_h[0]*clf_h[1]], dim=-1) if self.task_params.setdefault('concat_strategy', 'normal') == 'diff' else torch.cat(clf_h, dim=-1)
            clf_logits = self.linear(clf_h) if self.linear else clf_h
        else:
            clf_logits = clf_h = F.pairwise_distance(self.linear(clf_h[0]), self.linear(clf_h[1]), 2, eps=1e-12) if self.task_params['sentsim_func'] == 'dist' else F.cosine_similarity(self.linear(clf_h[0]), self.linear(clf_h[1]), dim=1, eps=1e-12)
        if self.thrshlder: self.thrshld = self.thrshlder(clf_h)
        if self.do_lastdrop: clf_logits = self.last_dropout(clf_logits)

        if (labels is None):
            if self.crf:
                tag_seq, score = zip(*self.crf.viterbi_tags(clf_logits.view(input_ids.size()[0], -1, self.num_lbs), torch.ones_like(input_ids)))
                tag_seq = torch.tensor(tag_seq).to('cuda') if use_gpu else torch.tensor(tag_seq)
                print((tag_seq.min(), tag_seq.max(), score))
                clf_logits = torch.zeros((*tag_seq.size(), self.num_lbs)).to('cuda') if use_gpu else torch.zeros((*tag_seq.size(), self.num_lbs))
                clf_logits = clf_logits.scatter(-1, tag_seq.unsqueeze(-1), 1)
                return clf_logits
            for cnstrnt in self.constraints: clf_logits = cnstrnt(clf_logits)
            if (self.mlt_trnsfmr and self.task_type in ['entlmnt', 'sentsim'] and self.task_params.setdefault('sentsim_func', None) is not None and self.task_params['sentsim_func'] != 'concat' and self.task_params['sentsim_func'] != self.task_params.setdefault('ymode', 'sim')): return 1 - clf_logits.view(-1, self.num_lbs)
            return clf_logits.view(-1, self.num_lbs)
        if self.crf:
            clf_loss = -self.crf(clf_logits.view(input_ids.size()[0], -1, self.num_lbs), pool_idx)
            if sample_weights is not None: clf_loss *= sample_weights
            return clf_loss, None
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
            loss_cls = C.RGRSN_LOSS_MAP[self.task_params.setdefault('loss', 'contrastive' if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else 'mse')]
            loss_func = loss_cls(reduction='none', x_mode=C.SIM_FUNC_MAP.setdefault(self.task_params['sentsim_func'], 'dist'), y_mode=self.task_params.setdefault('ymode', 'sim')) if self.task_params.setdefault('sentsim_func', None) and self.task_params['sentsim_func'] != 'concat' else (loss_cls(reduction='none', x_mode='sim', y_mode=self.task_params.setdefault('ymode', 'sim')) if self.task_params['sentsim_func'] == 'concat' else nn.MSELoss(reduction='none'))
            clf_loss = loss_func(clf_logits.view(-1), labels.view(-1))
        if self.thrshlder:
            num_lbs = labels.view(-1, self.num_lbs).sum(1)
            clf_loss = 0.8 * clf_loss + 0.2 * F.mse_loss(self.thrshld, torch.sigmoid(torch.topk(clf_logits, k=num_lbs.max(), dim=1, sorted=True)[0][:,num_lbs-1]), reduction='mean')
        if sample_weights is not None: clf_loss *= sample_weights
        return clf_loss, lm_loss
