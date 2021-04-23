#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2021 by Caspar. All rights reserved.
# File Name: .py
# Author: Shankai Yan
# E-mail: dr.skyan@gmail.com
# Created Time: 2021-03-29 17:19:30
###########################################################################
#

import numpy as np
from scipy.sparse import csc_matrix

import torch
from torch import nn
import torch.nn.functional as F

from . import transformer as T


class SiameseRankHead(nn.Module):
    def __init__(self, linear, siamese_linear, binlbr, tokenizer, encode_func, trnsfm, special_tknids_args, pad_val, base_model, thrshld_lnr=0.5, thrshld_sim=0.5, topk=None, num_sampling_note=1, lbnotes='lbnotes.csv'):
        super(SiameseRankHead, self).__init__()
        self.linear = linear
        self.siamese_linear = siamese_linear
        self.base_model = dict(zip(['model', 'tokenizer', 'encode_func', 'special_tknids_args', 'pad_val', 'transforms', 'transforms_args', 'transforms_kwargs'], [base_model, tokenizer, encode_func, special_tknids_args, pad_val]+trnsfm))
        self.binlbr = binlbr
        self.thrshld_lnr = thrshld_lnr
        self.thrshld_sim = thrshld_sim
        self.topk = topk
        self.num_sampling_note = num_sampling_note
        self.lbnotes = lbnotes if type(lbnotes) is pd.DataFrame else pd.read_csv(lbnotes, sep='\t', index_col='id', encoding='utf-8')

    def forward(self, hidden_states):
        # generate candidates with linear and rank the candidates with siamese_linear
        use_gpu = next(self.parameters()).is_cuda
        batch_size = 125
        logits = self.linear(hidden_states)
        prob = torch.sigmoid(logits)
        if self.topk:
            topk, topk_indices = torch.topk(prob.data, self.topk)
            prob_topk = torch.zeros(prob.size(), dtype=topk.dtype).scatter_(1, topk_indices.cpu(), topk.cpu())
            pred = (prob_topk > self.thrshld_lnr).int()
        else:
            pred = (prob.data > self.thrshld_lnr).int()
        pred_csc = csc_matrix(pred.cpu().numpy())
        pred_scores = [[] for x in range(len(pred_csc.data))]
        labels = [self.binlbr[i] for i in range(len(self.binlbr))]
        num_notes = np.array([min(self.num_sampling_note, self.lbnotes.loc[lb].shape[0]) if lb in self.lbnotes.index else 0 for lb in labels])
        num_samps = pred_csc.indptr[1:] - pred_csc.indptr[:-1]
        total_num_notes, total_num_pairs = sum(num_notes), sum(num_samps * num_notes)
        def gen_samples():
            for i, lb in enumerate(labels):
                for note in (self.lbnotes.loc[lb]['text'].sample(n=min(self.num_sampling_note, self.lbnotes.loc[lb].shape[0])) if type(self.lbnotes.loc[lb]['text']) is pd.Series else [self.lbnotes.loc[lb]['text']]) if lb in self.lbnotes.index else []:
                    tkns = self._transform_chain((self.base_model['encode_func'](note, self.base_model['tokenizer']), None), self.base_model['transforms'], self.base_model['transforms_args'], self.base_model['transforms_kwargs'])[0]
                    yield tkns, i, pred_csc.indices[pred_csc.indptr[i]:pred_csc.indptr[i+1]]
        def gen_pairs():
            batch_tkns, lb_indices, orig_indices = [[] for x in range(3)]
            for i, (x, lbidx, orig_idx) in enumerate(gen_samples()):
                batch_tkns.append(x)
                lb_indices.append(lbidx)
                orig_indices.append(orig_idx)
                if i > 0 and i % batch_size == 0 or i == total_num_notes - 1:
                    clf_tknids = self.base_model['special_tknids_args']['clf_tknids']
                    tkns_tnsr = torch.tensor(batch_tkns, dtype=torch.long).to('cuda') if use_gpu else torch.tensor(batch_tkns, dtype=torch.long)
                    mask_tnsr = (~tkns_tnsr.eq(self.base_model['pad_val'] * torch.ones_like(tkns_tnsr))).long()
                    pool_idx = tkns_tnsr.eq(clf_tknids[0] * torch.ones_like(tkns_tnsr)).int().argmax(-1)
                    note_clf_h = self.base_model['model'](tkns_tnsr, mask_tnsr if type(self.base_model['model']) is T.BERTClfHead else pool_idx, embedding_mode=True)
                    # note_clf_h = hidden_states[:1]
                    for j, clf_h in enumerate(note_clf_h):
                        for orig_id in orig_indices[j]:
                            yield clf_h, hidden_states[orig_id], lb_indices[j], orig_id
                    batch_tkns, lb_indices, orig_indices = [[] for x in range(3)]
        batch_pairs, indices = [[] for x in range(2)]
        for i, (h1, h2, lbidx, orig_id) in enumerate(tqdm(gen_pairs(), desc='[%i pair(s)] Siamese similarity ranking' % total_num_pairs)):
            batch_pairs.append([h1, h2])
            indices.append([lbidx, orig_id])
            if i > 0 and i % batch_size == 0 or i == total_num_pairs - 1:
                clf_h = [torch.stack(x) for x in zip(*batch_pairs)]
                continue
                if (self.base_model['model'].task_params.setdefault('sentsim_func', None) is None or self.base_model['model'].task_params['sentsim_func'] == 'concat'):
                    clf_h = torch.cat(clf_h, dim=-1)
                    sim_logits = self.siamese_linear(clf_h) if self.siamese_linear else clf_h
                else:
                    sim_logits = F.pairwise_distance(self.siamese_linear(clf_h[0]), self.siamese_linear(clf_h[1]), 2, eps=1e-12) if self.base_model['model'].task_params['sentsim_func'] == 'dist' else F.cosine_similarity(self.siamese_linear(clf_h[0]), self.siamese_linear(clf_h[1]), dim=1, eps=1e-12)
                sim_probs = torch.sigmoid(sim_logits).data
                for idx, sim_prob in zip(indices, sim_probs):
                    orig_idx = pred_csc.indptr[idx[0]], pred_csc.indptr[idx[0]+1]
                    pred_scores[orig_idx[0]:orig_idx[1]][np.where(pred_csc.indices[orig_idx[0]:orig_idx[1]]==idx[1])[0][0]].append(sim_prob.cpu().numpy())
                batch_pairs, indices = [[] for x in range(2)]
        pred_scores = [np.mean(x) for x in pred_scores]
        sim_probs_tnsr = torch.tensor(csc_matrix((pred_scores, pred_csc.indices, pred_csc.indptr), shape=pred_csc.shape).todense(), dtype=torch.float)
        if use_gpu: sim_probs_tnsr = sim_probs_tnsr.to('cuda')
        logits = sim_probs_tnsr * prob
        return logits

    def to(self, *args, **kwargs):
        super(SiameseRankHead, self).to(*args, **kwargs)
        self.linear = self.linear.to(*args, **kwargs)
        return self

    def _transform_chain(self, sample, transforms, transforms_args={}, transforms_kwargs={}):
        if transforms:
            transforms = transforms if type(transforms) is list else [transforms]
            transforms_kwargs = transforms_kwargs if type(transforms_kwargs) is list else [transforms_kwargs]
            for transform, transform_kwargs in zip(transforms, transforms_kwargs):
                transform_kwargs.update(transforms_args)
                sample = transform(sample, **transform_kwargs) if callable(transform) else getattr(self, transform)(sample, **transform_kwargs)
        return sample


class SiameseRankTransformer(object):
    def __init__(self, clf):
        self.clf = clf

    def merge_siamese(self, tokenizer, encode_func, trnsfm, special_tknids_args, pad_val=0, thrshld_lnr=0.5, thrshld_sim=0.5, topk=None, lbnotes='lbnotes.csv'):
        use_gpu = next(self.clf.parameters()).is_cuda
        self.clf.task_type = self.clf.clf_task_type
        self.clf.linear = SiameseRankHead(self.clf.clf_linear, self.clf.linear, self.clf.binlbr, tokenizer=tokenizer, encode_func=encode_func, trnsfm=trnsfm, special_tknids_args=special_tknids_args, pad_val=pad_val, base_model=self.clf, thrshld_lnr=thrshld_lnr, thrshld_sim=thrshld_sim, topk=topk, lbnotes=lbnotes)
        self.clf.linear = self.clf.linear.to('cuda') if use_gpu else self.clf.linear
        self.clf.mode = 'clf'
