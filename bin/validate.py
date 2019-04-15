#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: validate.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2019-03-29 19:27:12
###########################################################################
#

import os, sys, time, copy, pickle, logging, itertools
from collections import OrderedDict
from optparse import OptionParser
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from sklearn import metrics

from allennlp.modules.elmo import Elmo, batch_to_ids
from pytorch_pretrained_bert import OpenAIGPTConfig, OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, TransfoXLConfig, TransfoXLTokenizer, TransfoXLLMHeadModel

import spacy
nlp = spacy.load("en_core_web_sm")

from bionlp.util import io, system


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
SC=';;'

opts, args = {}, []
cfgr = None


class BaseClfHead(nn.Module):
    """ Classifier Head for the Basic Language Model """

    def __init__(self, lm_model, config, task_type, num_lbs=1, pdrop=0.1, mlt_trnsfmr=False, **kwargs):
        super(BaseClfHead, self).__init__()
        self.lm_model = lm_model
        self.task_type = task_type
        self.dropout = nn.Dropout2d(pdrop) if task_type == 'nmt' else nn.Dropout(pdrop)
        self.last_dropout = nn.Dropout(pdrop)
        self.lm_logit = self._mlt_lm_logit if mlt_trnsfmr else self._lm_logit
        self.clf_h = self._mlt_clf_h if mlt_trnsfmr else self._clf_h
        self.num_lbs = num_lbs
        self.kwprop = {}
        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(self, input_ids, pool_idx, labels=None, past=None, weights=None):
        use_gpu = next(self.parameters()).is_cuda
        trnsfm_output = self.transformer(input_ids)
        (hidden_states, past) = trnsfm_output if type(trnsfm_output) is tuple else (trnsfm_output, None)
        lm_logits, lm_target = self.lm_logit(input_ids=input_ids, hidden_states=hidden_states, past=past)
        lm_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        lm_loss = lm_loss_func(lm_logits.contiguous().view(-1, lm_logits.size(-1)), lm_target.contiguous().view(-1)).view(input_ids.size(0), -1)

        clf_h, pool_idx = self.clf_h(hidden_states, pool_idx)
        if self.task_type == 'nmt':
            clf_h = clf_h
        else:
            smp_offset = torch.arange(input_ids.size(0))
            smp_offset, pool_idx = (smp_offset.to('cuda'), pool_idx.to('cuda')) if (use_gpu) else (smp_offset, pool_idx)
            smp_offset = smp_offset.to('cuda') if use_gpu else smp_offset
            pool_offset = smp_offset * input_ids.size(-1) + pool_idx
            pool_h = pool_offset.unsqueeze(-1).expand(-1, self.n_embd)
            pool_h = pool_h.to('cuda') if use_gpu else pool_h
            clf_h = clf_h.gather(0, pool_h)
        clf_h = self.norm(clf_h)
        clf_h = self.dropout(clf_h)
        clf_logits = self.linear(clf_h)
        clf_logits = self.last_dropout(clf_logits)

        if (labels is None): return clf_logits.view(-1, self.num_lbs)
        if self.task_type == 'mltc-clf' or self.task_type == 'entlmnt' or self.task_type == 'nmt':
            loss_func = nn.CrossEntropyLoss(weight=weights, reduction='none')
            clf_loss = loss_func(clf_logits.view(-1, self.num_lbs), labels.view(-1))
        elif self.task_type == 'mltl-clf':
            loss_func = nn.MultiLabelSoftMarginLoss(weight=weights, reduction='none')
            clf_loss = loss_func(clf_logits.view(-1, self.num_lbs), labels.view(-1, self.num_lbs))
        elif self.task_type == 'sentsim':
            loss_func = nn.MSELoss(reduction='none')
            clf_loss = loss_func(clf_logits.view(-1), labels.view(-1))
        return clf_loss, lm_loss

    def _clf_h(self, hidden_states, pool_idx, past=None):
        return hidden_states.view(-1, self.n_embd), pool_idx

    def _mlt_clf_h(self, hidden_states, pool_idx, past=None):
        return hidden_states.sum(1).view(-1, self.n_embd), pool_idx.max(1)[0]

    def transformer(self, input_ids):
        return self.lm_model.transformer(input_ids=input_ids)

    def _lm_logit(self, input_ids, hidden_states, past=None):
        lm_h = hidden_states[:,:-1]
        return self.lm_model.lm_head(lm_h), input_ids[:,1:]

    def _mlt_lm_logit(self, input_ids, hidden_states, past=None):
        lm_h = hidden_states[:,:,:-1].contiguous().view(-1, self.n_embd)
        lm_target = input_ids[:,:,1:].contiguous().view(-1)
        return self.lm_model.lm_head(lm_h), lm_target.view(-1)


class GPTClfHead(BaseClfHead):
    def __init__(self, lm_model, config, task_type, num_lbs=1, pdrop=0.1, mlt_trnsfmr=False, **kwargs):
        super(GPTClfHead, self).__init__(lm_model, config, task_type, num_lbs=num_lbs, pdrop=pdrop, mlt_trnsfmr=mlt_trnsfmr, **kwargs)
        if type(lm_model) is GPT2LMHeadModel:self.kwprop['past_paramname'] = 'past'
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.norm = nn.BatchNorm1d(config.n_embd)
        self._int_actvtn = nn.ReLU
        self._out_actvtn = nn.Sigmoid
        self.linear = nn.Linear(config.n_embd, num_lbs)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)
        # self.linear = nn.Sequential(self.linear, nn.Sigmoid()) if task_type == 'sentsim' else self.linear
        self.linear = nn.Sequential(nn.Linear(config.n_embd, 1024), self._int_actvtn(), nn.Linear(1024, num_lbs), self._out_actvtn()) if task_type == 'sentsim' else self.linear


class TransformXLClfHead(BaseClfHead):
    def __init__(self, lm_model, config, task_type, num_lbs=1, pdrop=0.1, mlt_trnsfmr=False, **kwargs):
        super(TransformXLClfHead, self).__init__(lm_model, config, task_type, num_lbs=num_lbs, pdrop=pdrop, mlt_trnsfmr=mlt_trnsfmr, **kwargs)
        self.kwprop['past_paramname'] = 'mems'
        self.vocab_size = config.n_token
        self.n_embd = config.d_embed
        self.norm = nn.BatchNorm1d(config.n_embd)
        self.linear = nn.Linear(config.d_embed, num_lbs)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def _lm_logit(self, input_ids, hidden_states, past=None):
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

    def _mlt_lm_logit(self, input_ids, hidden_states, past=None):
        return self._lm_logit(input_ids, hidden_states, past=past)

    def transformer(self, input_ids):
        return self.lm_model.transformer(input_ids=input_ids.view(input_ids.size(0), -1))


class ELMoClfHead(BaseClfHead):
    def __init__(self, lm_model, config, task_type, hidden_dim=768, num_lbs=1, pdrop=0.1, mlt_trnsfmr=False, maxpool=False, **kwargs):
        super(ELMoClfHead, self).__init__(lm_model, config, task_type, num_lbs=num_lbs, pdrop=pdrop, mlt_trnsfmr=mlt_trnsfmr, **kwargs)
        self.vocab_size = 793471
        self.n_embd = 1024 * (4 if task_type == 'sentsim' else 2)
        self._int_actvtn = nn.ReLU
        self._out_actvtn = nn.Sigmoid
        if task_type == 'nmt':
            self.maxpool = None
            self.norm = nn.BatchNorm1d(128)
            if (hidden_dim is None):
                self.linear = nn.Linear(self.n_embd, num_lbs)
                nn.init.normal_(self.linear.weight, std=0.02)
                nn.init.normal_(self.linear.bias, 0)
            else:
                self.linear = nn.Sequential(nn.Linear(self.n_embd, hidden_dim), self._int_actvtn(), nn.Linear(hidden_dim, hidden_dim), self._int_actvtn(), nn.Linear(hidden_dim, num_lbs), self._out_actvtn()) if self.task_type == 'sentsim' else nn.Sequential(nn.Linear(self.n_embd, hidden_dim), self._int_actvtn(), nn.Linear(hidden_dim, hidden_dim), self._int_actvtn(), nn.Linear(hidden_dim, num_lbs))
        elif maxpool:
            self.maxpool = nn.MaxPool2d(8, stride=4)
            self.norm = nn.BatchNorm1d(32130 if self.task_type == 'sentsim' or self.task_type == 'entlmnt' else 16065)
            if (hidden_dim is None):
                self.linear = nn.Linear(32130 if self.task_type == 'sentsim' or self.task_type == 'entlmnt' else 16065, num_lbs)
                nn.init.normal_(self.linear.weight, std=0.02)
                nn.init.normal_(self.linear.bias, 0)
            else:
                self.linear = nn.Sequential(nn.Linear(32130, hidden_dim), self._int_actvtn(), nn.Linear(hidden_dim, hidden_dim), self._int_actvtn(), nn.Linear(hidden_dim, num_lbs), self._out_actvtn()) if self.task_type == 'sentsim' else nn.Sequential(nn.Linear(32130 if self.task_type == 'entlmnt' or self.task_type == 'sentsim' else 16065, hidden_dim), self._int_actvtn(), nn.Linear(hidden_dim, hidden_dim), self._int_actvtn(), nn.Linear(hidden_dim, num_lbs))
        else:
            self.maxpool = None
            self.norm = nn.BatchNorm1d(self.n_embd)
            if (hidden_dim is None):
                self.linear = nn.Linear(self.n_embd, num_lbs)
            else:
                self.linear = nn.Sequential(nn.Linear(self.n_embd, hidden_dim), self._int_actvtn(), nn.Linear(hidden_dim, hidden_dim), self._int_actvtn(), nn.Linear(hidden_dim, num_lbs), self._out_actvtn()) if self.task_type == 'sentsim' else nn.Sequential(nn.Linear(self.n_embd, hidden_dim), self._int_actvtn(), nn.Linear(hidden_dim, hidden_dim), self._int_actvtn(), nn.Linear(hidden_dim, num_lbs))


    def forward(self, input_ids, pool_idx, labels=None, past=None, weights=None):
        if self.task_type == 'entlmnt' or self.task_type == 'sentsim':
            embeddings = (self.lm_model(input_ids[0]), self.lm_model(input_ids[1]))
            clf_h = torch.cat(embeddings[0]['elmo_representations'], dim=-1), torch.cat(embeddings[1]['elmo_representations'], dim=-1)
            if self.maxpool:
                clf_h = [clf_h[x].view(clf_h[x].size(0), 2*clf_h[x].size(1), -1) for x in [0,1]]
                clf_h = [self.maxpool(clf_h[x]).view(clf_h[x].size(0), -1) for x in [0,1]]
            else:
                clf_h = [clf_h[x].gather(1, pool_idx[x].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h[x].size(2))).squeeze(1) for x in [0,1]]
            clf_h = torch.cat(clf_h, dim=-1) if self.task_type == 'entlmnt' else torch.cat(clf_h, dim=-1) + torch.cat(clf_h[::-1], dim=-1)
        else:
            embeddings = self.lm_model(input_ids)
            clf_h = torch.cat(embeddings['elmo_representations'], dim=-1)
            if self.task_type == 'nmt':
                clf_h = clf_h
            elif self.maxpool:
                clf_h = clf_h.view(clf_h.size(0), 2*clf_h.size(1), -1)
                clf_h = self.maxpool(clf_h).view(clf_h.size(0), -1)
            else:
                clf_h = clf_h.gather(1, pool_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, clf_h.size(2))).squeeze(1)
        clf_h = self.norm(clf_h)
        clf_h = self.dropout(clf_h)
        clf_logits = self.linear(clf_h)
        clf_logits = self.last_dropout(clf_logits)

        if (labels is None): return clf_logits.view(-1, self.num_lbs)
        if self.task_type == 'mltc-clf' or self.task_type == 'entlmnt' or self.task_type == 'nmt':
            loss_func = nn.CrossEntropyLoss(weight=weights, reduction='none')
            clf_loss = loss_func(clf_logits.view(-1, self.num_lbs), labels.view(-1))
        elif self.task_type == 'mltl-clf':
            loss_func = nn.MultiLabelSoftMarginLoss(weight=weights, reduction='none')
            clf_loss = loss_func(clf_logits.view(-1, self.num_lbs), labels.view(-1, self.num_lbs))
        elif self.task_type == 'sentsim':
            loss_func = nn.MSELoss(reduction='none')
            clf_loss = loss_func(clf_logits.view(-1), labels.view(-1))
        return clf_loss, None



class BaseDataset(Dataset):
    """Basic dataset class"""

    def __init__(self, csv_file, text_col, label_col, encode_func, tokenizer, sep='\t', binlb=None, transforms=[], transforms_args={}, transforms_kwargs=[], **kwargs):
        self.df = self._df = csv_file if type(csv_file) is pd.DataFrame else pd.read_csv(csv_file, sep=sep, encoding='utf-8', engine='python', error_bad_lines=False, **kwargs)
        self.df.columns = self.df.columns.astype(str, copy=False)
        self.text_col = [str(s) for s in text_col] if hasattr(text_col, '__iter__') and type(text_col) is not str else str(text_col)
        self.label_col = [str(s) for s in label_col] if hasattr(label_col, '__iter__') and type(label_col) is not str else str(label_col)
        self.df = self.df[self.df[self.label_col].notnull()]
        if (binlb == 'rgrsn'):
            self.binlb = None
            self.binlbr = None
        elif (binlb is None):
            lb_df = self.df[self.df[self.label_col].notnull()][self.label_col]
            labels = sorted(set(lb_df)) if type(lb_df.iloc[0]) is not list else sorted(set([lb for lbs in lb_df for lb in lbs]))
            self.binlb = OrderedDict([(lb, i) for i, lb in enumerate(labels)])
            self.binlbr = OrderedDict([(i, lb) for i, lb in enumerate(labels)])
        else:
            self.binlb = binlb
            self.binlbr = OrderedDict([(i, lb) for lb, i in binlb.items()])
        self.encode_func = encode_func
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.transforms_args = transforms_args
        self.transforms_kwargs = transforms_kwargs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = self.encode_func(record[self.text_col], self.tokenizer), record[self.label_col]
        sample = self._transform_chain(sample)
        return self.df.index[idx], (sample[0] if type(sample[0]) is str or type(sample[0][0]) is str else torch.tensor(sample[0])), torch.tensor(sample[1])

    def _transform_chain(self, sample):
        if self.transforms:
            self.transforms = self.transforms if type(self.transforms) is list else [self.transforms]
            self.transforms_kwargs = self.transforms_kwargs if type(self.transforms_kwargs) is list else [self.transforms_kwargs]
            for transform, transform_kwargs in zip(self.transforms, self.transforms_kwargs):
                transform_kwargs.update(self.transforms_args)
                sample = transform(sample, **transform_kwargs) if callable(transform) else getattr(self, transform)(sample, **transform_kwargs)
        return sample

    def _nmt_transform(self, sample, options=None, binlb={}):
        if (len(binlb) > 0): self.binlb = binlb
        return sample[0], [self.binlb.setdefault(y, len(self.binlb)) for y in sample[1]]

    def _mltc_transform(self, sample, options=None, binlb={}):
        if (len(binlb) > 0): self.binlb = binlb
        return sample[0], self.binlb.setdefault(sample[1], len(self.binlb))

    def _mltl_transform(self, sample, options=None, binlb={}, get_lb=lambda x: x.split(SC)):
        if (len(binlb) > 0): self.binlb = binlb
        labels = get_lb(sample[1])
        return sample[0], [1. if lb in labels else 0. for lb in self.binlb.keys()]

    def fill_labels(self, lbs, binlb=True, index=None, saved_path=None, **kwargs):
        if binlb and self.binlbr is not None:
            lbs = [self.binlbr[lb] for lb in lbs]
        filled_df = self.df.copy(deep=True)
        if index:
            filled_df.loc[index][self.label_col] = lbs
        else:
            filled_df[self.label_col] = lbs
        if (saved_path is not None):
            filled_df.to_csv(saved_path, **kwargs)
        return filled_df

    def rebalance(self):
        if (self.binlb is None): return
        task_cols, task_trsfm, task_extparms = TASK_COL_MAP[opts.task], TASK_TRSFM[opts.task], TASK_EXT_PARAMS[opts.task]
        lb_trsfm = [x['get_lb'] for x in task_trsfm[1] if 'get_lb' in x]
        self.df = self._df
        if len(lb_trsfm) > 0:
            lb_df = self.df[task_cols['y']].apply(lb_trsfm[0])
        else:
            lb_df = self.df[task_cols['y']]
        if (type(lb_df.iloc[0]) is list):
            lb_df[:] = [self._mltl_transform((None, SC.join(lbs)))[1] for lbs in lb_df]
            max_lb_df = lb_df.loc[[idx for idx, lbs in lb_df.iteritems() if np.sum(map(int, lbs)) == 0]]
            max_num, avg_num = max_lb_df.shape[0], 1.0 * lb_df[~max_lb_df].shape[0] / len(lb_df.iloc[0])
        else:
            class_count = np.array([[1 if lb in y else 0 for lb in self.binlb.keys()] for y in lb_df if y is not None]).sum(axis=0)
            max_num, max_lb_bin = class_count.max(), class_count.argmax()
            max_lb_df = lb_df[lb_df == self.binlbr[max_lb_bin]]
            avg_num = np.mean([class_count[x] for x in range(len(class_count)) if x != max_lb_bin])
        removed_idx = max_lb_df.sample(n=int(max_num-avg_num), random_state=1).index
        self.df = self.df.loc[list(set(self.df.index)-set(removed_idx))]

    def remove_mostfrqlb(self):
        if (self.binlb is None or self.binlb == 'rgrsn'): return
        task_cols, task_trsfm, task_extparms = TASK_COL_MAP[opts.task], TASK_TRSFM[opts.task], TASK_EXT_PARAMS[opts.task]
        lb_trsfm = [x['get_lb'] for x in task_trsfm[1] if 'get_lb' in x]
        self.df = self._df
        if len(lb_trsfm) > 0:
            lb_df = self.df[task_cols['y']].apply(lb_trsfm[0])
        else:
            lb_df = self.df[task_cols['y']]
        class_count = np.array([[1 if lb in y else 0 for lb in self.binlb.keys()] for y in lb_df if y]).sum(axis=0)
        max_num, max_lb_bin = class_count.max(), class_count.argmax()
        max_lb_df = lb_df[lb_df == self.binlbr[max_lb_bin]]
        self.df = self.df.loc[list(set(self.df.index)-set(max_lb_df.index))]


class SentSimDataset(BaseDataset):
    """Sentence Similarity task dataset class"""

    def __init__(self, csv_file, text_col, label_col, encode_func, tokenizer, sep='\t', binlb=None, transforms=[], transforms_args={}, transforms_kwargs=[], **kwargs):
        super(SentSimDataset, self).__init__(csv_file, text_col, label_col, encode_func, tokenizer, sep=sep, binlb=binlb, transforms=transforms, transforms_args=transforms_args, transforms_kwargs=transforms_kwargs, **kwargs)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = [self.encode_func(record[sent_idx], self.tokenizer) for sent_idx in self.text_col], record[self.label_col]
        sample = self._transform_chain(sample)
        return self.df.index[idx], (sample[0] if type(sample[0][0]) is str or type(sample[0][0][0]) is str else torch.tensor(sample[0])), torch.tensor(sample[1] / 5.0)

    def fill_labels(self, lbs, index=None, saved_path=None, **kwargs):
        lbs = 5.0 * lbs
        filled_df = self.df.copy(deep=True)
        if index:
            filled_df.loc[index][self.label_col] = lbs
        else:
            filled_df[self.label_col] = lbs
        if (saved_path is not None):
            filled_df.to_csv(saved_path, **kwargs)
        return filled_df


class EntlmntDataset(BaseDataset):
    """Entailment task dataset class"""

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = [self.encode_func(record[sent_idx], self.tokenizer) for sent_idx in self.text_col], record[self.label_col]
        sample = self._transform_chain(sample)
        return self.df.index[idx], (sample[0] if type(sample[0][0]) is str or (type(sample[0][0]) is list and type(sample[0][0][0]) is str) else torch.tensor(sample[0])), torch.tensor(sample[1])


class NERDataset(BaseDataset):
    """NER task dataset class"""

    def __init__(self, csv_file, text_col, label_col, encode_func, tokenizer, sep='\t', binlb=None, transforms=[], transforms_args={}, transforms_kwargs=[], **kwargs):
        records = []
        with open(csv_file, 'r', encoding='utf-8') as fd:
            for line in fd.readlines():
                line = line.strip().strip('\n').split('\t')
                if (len(line) == 0 or line[0] == '' or line[0].isspace() or (len(records) > 0 and records[-1] == '.' and line[0] == '.')): continue
                records.append(line)
        df = pd.DataFrame(records, columns=[text_col, 'a', 'b', label_col])
        super(NERDataset, self).__init__(df, text_col, label_col, encode_func, tokenizer, sep=sep, header=None, binlb=binlb, transforms=transforms, transforms_args=transforms_args, transforms_kwargs=transforms_kwargs, **kwargs)
        # _ = self.df[self.text_col].replace('', np.nan, inplace=True)
        # self.df.dropna(subset=[self.text_col])
        texts = self.df.iloc[:,self.text_col] if type(self.text_col) is int else self.df[self.text_col]
        sep_selector = texts.apply(lambda x: True if x=='.' else False)
        int_idx = pd.DataFrame(np.arange(self.df.shape[0]), index=self.df.index)
        self.boundaries = [0] + list(itertools.chain.from_iterable((int_idx[sep_selector.values].values+1).tolist()))

    def __len__(self):
        return len(self.boundaries) - 1

    def __getitem__(self, idx):
        record = self.df.iloc[self.boundaries[idx]:self.boundaries[idx+1]]
        sample = self.encode_func(record[self.text_col].values.tolist(), self.tokenizer), record[self.label_col].values.tolist()
        num_samples = [len(x) for x in sample[0]] if (len(sample[0]) > 0 and type(sample[0][0]) is list) else [1] * len(sample[0])
        record_idx = [0] + np.cumsum(num_samples).tolist()
        is_empty = (type(sample[0]) is list and len(sample[0]) == 0) or (type(sample[0]) is list and len(sample[0]) > 0 and type(sample[0][0]) is list and len(sample[0][0]) == 0)
        if (is_empty): return SC.join(map(str, self.df.index[self.boundaries[idx]:self.boundaries[idx+1]].values.tolist())), '' if self.encode_func == _tokenize else torch.LongTensor([-1]*opts.maxlen), '' if self.encode_func == _tokenize else torch.LongTensor([-1]*opts.maxlen), SC.join(map(str, record_idx))
        is_encoded = (type(sample[0]) is list and type(sample[0][0]) is int) or (type(sample[0]) is list and len(sample[0]) > 0 and type(sample[0][0]) is list and len(sample[0][0]) > 0 and type(sample[0][0][0]) is int)
        sample = list(itertools.chain.from_iterable(sample[0])) if is_encoded else sample[0], list(itertools.chain.from_iterable([[x] * ns for x, ns in zip(sample[1], num_samples)]))
        sample = self._transform_chain(sample)
        return SC.join(map(str, self.df.index[self.boundaries[idx]:self.boundaries[idx+1]].values.tolist())), (torch.tensor(sample[0]) if is_encoded else SC.join(sample[0])), (torch.tensor(sample[1]) if is_encoded else SC.join(map(str, sample[1]))), SC.join(map(str, record_idx))

    def fill_labels(self, lbs, saved_path=None, binlb=True, index=None, **kwargs):
        if binlb and self.binlbr is not None:
            lbs = [self.binlbr[lb] for lb in lbs]
        filled_df = self.df.copy(deep=True)
        if index:
            filled_df.loc[index][self.label_col] = lbs
        else:
            filled_df[self.label_col] = lbs
        if (saved_path is not None):
            content = ['\t'.join(['index']+list(map(str, self.df.columns.tolist())))]
            for idx, row in filled_df.iterrows():
                content.append('\t'.join([str(idx)]+list(map(str, row.values.tolist()))))
                if (row[self.label_col] == '.'): content.append('\n')
            content = '\n'.join(content)
            with open(saved_path, 'w') as fd:
                fd.write(content)
        return filled_df


def _sentclf_transform(sample, options=None, start_tknids=[], clf_tknids=[]):
    X, y = sample
    X = [start_tknids + x + clf_tknids for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else start_tknids + X + clf_tknids
    return X, y


def _entlmnt_transform(sample, options=None, start_tknids=[], clf_tknids=[], delim_tknids=[]):
    X, y = sample
    X = start_tknids + X[0] + delim_tknids + X[1] + clf_tknids
    return X, y


def _sentsim_transform(sample, options=None, start_tknids=[], clf_tknids=[], delim_tknids=[]):
    X, y = sample
    X = [start_tknids + X[0] + delim_tknids + X[1] + clf_tknids, start_tknids + X[1] + delim_tknids + X[0] + clf_tknids]
    return X, y


def _padtrim_transform(sample, options=None, seqlen=32, xpad_val=0, ypad_val=None):
    X, y = sample
    X = [x[:min(seqlen, len(x))] + [xpad_val] * (seqlen - len(x)) for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else X[:min(seqlen, len(X))] + [xpad_val] * (seqlen - len(X))
    if ypad_val is not None: y = [x[:min(seqlen, len(x))] + [ypad_val] * (seqlen - len(x)) for x in y] if hasattr(y, '__iter__') and len(y) > 0 and type(y[0]) is not str and hasattr(y[0], '__iter__') else y[:min(seqlen, len(y))] + [ypad_val] * (seqlen - len(y))
    return X, y


def _adjust_encoder(mdl_name, tokenizer, extra_tokens=[], ret_list=False):
    encoded_extknids = []
    if (mdl_name == 'gpt'):
        for tkn in extra_tokens:
            # tokenizer.encoder[tkn] = len(tokenizer.encoder)
            # encoded_extknids.append([tokenizer.encoder[tkn]] if ret_list else tokenizer.encoder[tkn])
            encoded_extknids.append([tokenizer.convert_tokens_to_ids(tkn)] if ret_list else tokenizer.convert_tokens_to_ids(tkn))
    elif (mdl_name == 'gpt2'):
        encoded_extknids = []
        for tkn in extra_tokens:
            tkn_ids = tokenizer.encode(tkn)
            encoded_extknids.append([tkn_ids] if (ret_list and type(tkn_ids) is not list) else tkn_ids)
    elif (mdl_name == 'trsfmxl'):
        for tkn in extra_tokens:
            tokenizer.__dict__[tkn] = len(tokenizer.__dict__)
            encoded_extknids.append([tokenizer.__dict__[tkn]] if ret_list else tokenizer.__dict__[tkn])
    else:
        encoded_extknids = [None] * len(extra_tokens)
    return encoded_extknids


def _gpt_encode(text, tokenizer):
    texts, records = [text] if (type(text) is str) else text, []
    try:
        for txt in texts:
            tokens = tokenizer.tokenize(txt)
            record = []
            while (len(tokens) > 512):
               record.extend(tokenizer.convert_tokens_to_ids(tokens[:512]))
               tokens = tokens[512:]
            record.extend(tokenizer.convert_tokens_to_ids(tokens))
            records.append(record)
    except Exception as e:
        print(e)
        print('Cannot encode %s' % str(text))
        return []
    return records[0] if (type(text) is str) else records


def _gpt2_encode(text, tokenizer):
    try:
        records = tokenizer.encode(text) if (type(text) is str) else [tokenizer.encode(line) for line in text]
    except Exception as e:
        print(e)
        print('Cannot encode %s' % str(text))
        return []
    return records


def _tokenize(text, tokenizer):
    return text
    try:
        records = [w.text for w in nlp(text)] if (type(text) is str) else [[w.text for w in nlp(line)] for line in text]
    except Exception as e:
        print(e)
        print('Cannot encode %s' % str(text))
        return []
    return records



def _onehot(y, size):
    y = torch.LongTensor(y).view(-1, 1)
    y_onehot = torch.FloatTensor(size[0], size[1])
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot.long()


def elmo_config(options_path, weights_path):
    return {'options_file':options_path, 'weight_file':weights_path, 'num_output_representations':2, 'dropout':opts.pdrop}

TASK_TYPE_MAP = {'bc5cdr-chem':'nmt', 'bc5cdr-dz':'nmt', 'shareclefe':'nmt', 'ddi':'mltc-clf', 'chemprot':'mltc-clf', 'i2b2':'mltc-clf', 'hoc':'mltl-clf', 'mednli':'entlmnt', 'biosses':'sentsim', 'clnclsts':'sentsim'}
TASK_PATH_MAP = {'bc5cdr-chem':'BC5CDR-chem', 'bc5cdr-dz':'BC5CDR-disease', 'shareclefe':'ShAReCLEFEHealthCorpus', 'ddi':'ddi2013-type', 'chemprot':'ChemProt', 'i2b2':'i2b2-2010', 'hoc':'hoc', 'mednli':'mednli', 'biosses':'BIOSSES', 'clnclsts':'clinicalSTS'}
TASK_DS_MAP = {'bc5cdr-chem':NERDataset, 'bc5cdr-dz':NERDataset, 'shareclefe':NERDataset, 'ddi':BaseDataset, 'chemprot':BaseDataset, 'i2b2':BaseDataset, 'hoc':BaseDataset, 'mednli':EntlmntDataset, 'biosses':SentSimDataset, 'clnclsts':SentSimDataset}
TASK_COL_MAP = {'bc5cdr-chem':{'index':False, 'X':'0', 'y':'1'}, 'bc5cdr-dz':{'index':False, 'X':'0', 'y':'1'}, 'shareclefe':{'index':False, 'X':'0', 'y':'1'}, 'ddi':{'index':'index', 'X':'sentence', 'y':'label'}, 'chemprot':{'index':'index', 'X':'sentence', 'y':'label'}, 'i2b2':{'index':'index', 'X':'sentence', 'y':'label'}, 'hoc':{'index':'index', 'X':'sentence', 'y':'labels'}, 'mednli':{'index':'id', 'X':['sentence1','sentence2'], 'y':'label'}, 'biosses':{'index':'index', 'X':['sentence1','sentence2'], 'y':'score'}, 'clnclsts':{'index':'index', 'X':['sentence1','sentence2'], 'y':'score'}}
# ([in_func|*], [in_func_params|*], [out_func|*], [out_func_params|*])
TASK_TRSFM = {'bc5cdr-chem':(['_nmt_transform'], [{}]), 'bc5cdr-dz':(['_nmt_transform'], [{}]), 'shareclefe':(['_nmt_transform'], [{}]), 'ddi':(['_mltc_transform'], [{}]), 'chemprot':(['_mltc_transform'], [{}]), 'i2b2':(['_mltc_transform'], [{}]), 'hoc':(['_mltl_transform'], [{ 'get_lb':lambda x: [s.split('_')[0] for s in x.split(',') if s.split('_')[1] == '1'], 'binlb': dict([(str(x),x) for x in range(10)])}]), 'mednli':(['_mltc_transform'], [{}]), 'biosses':([], []), 'clnclsts':([], [])}
TASK_EXT_TRSFM = {'bc5cdr-chem':([_padtrim_transform], [{}]), 'bc5cdr-dz':([_padtrim_transform], [{}]), 'shareclefe':([_padtrim_transform], [{}]), 'ddi':([_sentclf_transform, _padtrim_transform], [{},{}]), 'chemprot':([_sentclf_transform, _padtrim_transform], [{},{}]), 'i2b2':([_sentclf_transform, _padtrim_transform], [{},{}]), 'hoc':([_sentclf_transform, _padtrim_transform], [{},{}]), 'mednli':([_entlmnt_transform, _padtrim_transform], [{},{}]), 'biosses':([_sentsim_transform, _padtrim_transform], [{},{}]), 'clnclsts':([_sentsim_transform, _padtrim_transform], [{},{}])}
TASK_EXT_PARAMS = {'bc5cdr-chem':{'ypad_val':'O'}, 'bc5cdr-dz':{'ypad_val':'O'}, 'shareclefe':{'ypad_val':'O'}, 'ddi':{}, 'chemprot':{}, 'i2b2':{}, 'hoc':{'binlb': OrderedDict([(str(x),x) for x in range(10)])}, 'mednli':{}, 'biosses':{'binlb':'rgrsn'}, 'clnclsts':{'binlb':'rgrsn'}}

MDL_NAME_MAP = {'gpt2':'gpt2', 'gpt':'openai-gpt', 'trsfmxl':'transfo-xl-wt103', 'elmo':'elmo'}
PARAMS_MAP = {'gpt2':'GPT-2', 'gpt':'GPT', 'trsfmxl':'TransformXL', 'elmo':'ELMo'}
ENCODE_FUNC_MAP = {'gpt2':_gpt2_encode, 'gpt':_gpt_encode, 'trsfmxl':_gpt_encode, 'elmo':_tokenize}
MODEL_MAP = {'gpt2':GPT2LMHeadModel, 'gpt':OpenAIGPTLMHeadModel, 'trsfmxl':TransfoXLLMHeadModel, 'elmo':Elmo}
CLF_MAP = {'gpt2':GPTClfHead, 'gpt':GPTClfHead, 'trsfmxl':TransformXLClfHead, 'elmo':ELMoClfHead}
CLF_EXT_PARAMS = {'elmo':{'maxpool':False}}
CONFIG_MAP = {'gpt2':GPT2Config, 'gpt':OpenAIGPTConfig, 'trsfmxl':TransfoXLConfig, 'elmo':elmo_config}
TKNZR_MAP = {'gpt2':GPT2Tokenizer, 'gpt':OpenAIGPTTokenizer, 'trsfmxl':TransfoXLTokenizer, 'elmo':None}


def gen_mdl(mdl_name, pretrained=True, use_gpu=False, distrb=False, dev_id=None):
    if (type(pretrained) is str):
        print('Using pretrained model from `%s`' % pretrained)
        checkpoint = torch.load(pretrained, map_location='cpu')
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
    elif (pretrained):
        print('Using pretrained model...')
        model = MODEL_MAP[mdl_name].from_pretrained(MDL_NAME_MAP[mdl_name])
    else:
        print('Using untrained model...')
        try:
            common_cfg = cfgr('validate', 'common')
            pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
            params = pr('LM', PARAMS_MAP[mdl_name])
            config = CONFIG_MAP[mdl_name](**params)
            if (mdl_name == 'elmo'):
                pos_params = [config[k] for k in ['options_file','weight_file', 'num_output_representations']]
                kw_params = dict([(k, config[k]) for k in config.keys() if k not in ['options_file','weight_file', 'num_output_representations']])
                model = MODEL_MAP[mdl_name](*pos_params, **kw_params)
            else:
                model = MODEL_MAP[mdl_name](config)
        except Exception as e:
            print(e)
            print('Cannot find the pretrained model file, using online model instead.')
            model = MODEL_MAP[mdl_name].from_pretrained(MDL_NAME_MAP[mdl_name])
    if (use_gpu):
        if (distrb):
            if (type(dev_id) is list):
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=dev_id)
            else:
                torch.cuda.set_device(dev_id)
                model = model.cuda(dev_id)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev_id])
                raise NotImplementedError
                # Not implemented, should divide data outside this function and get one portion here, probably with parallel version of `load_data`
        elif (dev_id is not None):
            if (type(dev_id) is list):
                model.cuda()
                model = torch.nn.DataParallel(model, device_ids=dev_id)
            else:
                torch.cuda.set_device(dev_id)
                model = model.cuda(dev_id)
    return model


def gen_clf(mdl_name, use_gpu=False, distrb=False, dev_id=None, **kwargs):
    common_cfg = cfgr('validate', 'common')
    pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
    params = pr('LM', PARAMS_MAP[mdl_name])
    kwargs['config'] = CONFIG_MAP[mdl_name](**params)
    clf = CLF_MAP[mdl_name](**kwargs)
    return clf.to('cuda') if use_gpu else clf


def classify(dev_id=None):
    use_gpu = dev_id is not None
    encode_func = ENCODE_FUNC_MAP[opts.model]
    tokenizer = TKNZR_MAP[opts.model].from_pretrained(MDL_NAME_MAP[opts.model]) if TKNZR_MAP[opts.model] else None
    task_type = TASK_TYPE_MAP[opts.task]
    special_tkns = (['start_tknids', 'delim_tknids', 'clf_tknids'], ['_@_', ' _#_', ' _$_']) if task_type == 'sentsim' else (['start_tknids', 'clf_tknids'], ['_@_', ' _$_'])
    special_tknids = _adjust_encoder(opts.model, tokenizer, special_tkns[1], ret_list=True)
    special_tknids_args = dict(zip(special_tkns[0], special_tknids))

    # Prepare data
    task_path, task_dstype, task_cols, task_trsfm, task_extrsfm, task_extparms = TASK_PATH_MAP[opts.task], TASK_DS_MAP[opts.task], TASK_COL_MAP[opts.task], TASK_TRSFM[opts.task], TASK_EXT_TRSFM[opts.task], TASK_EXT_PARAMS[opts.task]
    trsfms = ([] if opts.model=='elmo' else task_extrsfm[0]) + (task_trsfm[0] if len(task_trsfm) > 0 else [])
    trsfms_kwargs = ([] if opts.model=='elmo' else ([{'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}] if TASK_TYPE_MAP[opts.task]=='nmt' else [special_tknids_args, {'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}])) + (task_trsfm[1] if len(task_trsfm) >= 2 else [{}] * len(task_trsfm))

    train_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'train.tsv'), task_cols['X'], task_cols['y'], ENCODE_FUNC_MAP[opts.model], tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms else None, transforms=trsfms, transforms_kwargs=trsfms_kwargs)
    lb_trsfm = [x['get_lb'] for x in task_trsfm[1] if 'get_lb' in x]
    if (task_type == 'sentsim'):
        class_count = None
    elif len(lb_trsfm) > 0:
        lb_df = train_ds.df[task_cols['y']].apply(lb_trsfm[0])
        class_count = np.array([[1 if lb in y else 0 for lb in task_extparms.setdefault('binlb', train_ds.binlb).keys()] for y in lb_df]).sum(axis=0)
    else:
        lb_df = train_ds.df[task_cols['y']]
        binlb = task_extparms.setdefault('binlb', train_ds.binlb)
        class_count = lb_df.value_counts()[binlb.keys()].values
    if (class_count is None):
        class_weights = None
        sampler = None
    else:
        class_weights = torch.Tensor(1.0 / class_count)
        class_weights /= class_weights.sum()
        sampler = WeightedRandomSampler(weights=class_weights, num_samples=opts.bsize, replacement=True)
    train_loader = DataLoader(train_ds, batch_size=opts.bsize, shuffle=False, sampler=None, num_workers=opts.np, drop_last=opts.droplast)

    dev_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'dev.tsv'), task_cols['X'], task_cols['y'], ENCODE_FUNC_MAP[opts.model], tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms else train_ds.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs)
    dev_loader = DataLoader(dev_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)
    test_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'test.tsv'), task_cols['X'], task_cols['y'], ENCODE_FUNC_MAP[opts.model], tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms else train_ds.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs)
    test_loader = DataLoader(test_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)

    # Build model
    lm_model = gen_mdl(opts.model, pretrained=True if type(opts.pretrained) is str and opts.pretrained.lower() == 'true' else opts.pretrained, use_gpu=use_gpu, distrb=opts.distrb, dev_id=dev_id)
    clf = gen_clf(opts.model, lm_model=lm_model, task_type=task_type, num_lbs=len(train_ds.binlb) if train_ds.binlb else 1, pdrop=opts.pdrop, mlt_trnsfmr=True if task_type=='sentsim' else False, use_gpu=use_gpu, distrb=opts.distrb, dev_id=dev_id, **dict([(k, getattr(opts, k)) if hasattr(opts, k) else (k, v) for k, v in CLF_EXT_PARAMS.setdefault(opts.model, {}).items()]))
    # optimizer = torch.optim.SGD(clf.parameters(), lr=opts.lr, momentum=0.9)
    optimizer = torch.optim.Adam(clf.parameters(), lr=opts.lr, weight_decay=opts.wdecay)

    # Training
    mdl_name = opts.model.lower().replace(' ', '_')
    train(clf, optimizer, train_loader, special_tknids_args['clf_tknids'], pad_val=train_ds.binlb[task_extparms.setdefault('ypad_val', 0)] if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), weights=class_weights, lmcoef=opts.lmcoef, clipmaxn=opts.clipmaxn, epochs=opts.epochs, task_type=task_type, task_name=opts.task, mdl_name=mdl_name, use_gpu=use_gpu, devq=dev_id)

    # Evaluation
    eval(clf, dev_loader, dev_ds.binlbr, special_tknids_args['clf_tknids'], pad_val=train_ds.binlb[task_extparms.setdefault('ypad_val', 0)] if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='dev', mdl_name=mdl_name, use_gpu=use_gpu)
    eval(clf, test_loader, test_ds.binlbr, special_tknids_args['clf_tknids'], pad_val=train_ds.binlb[task_extparms.setdefault('ypad_val', 0)] if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='test', mdl_name=mdl_name, use_gpu=use_gpu)


def train(clf, optimizer, dataset, clf_tknids, pad_val=0, weights=None, lmcoef=0.5, clipmaxn=0.25, epochs=1, task_type='mltc-clf', task_name='classification', mdl_name='sota', use_gpu=False, devq=None):
    clf.train()
    for epoch in range(epochs):
        total_loss = 0
        if task_type != 'entlmnt' and task_type != 'sentsim': dataset.dataset.rebalance()
        for step, batch in enumerate(tqdm(dataset, desc='[%i/%i epoch(s)] Training batches' % (epoch + 1, epochs))):
            optimizer.zero_grad()
            if task_type == 'nmt':
                idx, tkns_tnsr, lb_tnsr, record_idx = batch
                record_idx = [list(map(int, x.split(SC))) for x in record_idx]
            else:
                idx, tkns_tnsr, lb_tnsr = batch
            if (mdl_name == 'elmo'):
                if task_type == 'entlmnt' or task_type == 'sentsim':
                    tkns_tnsr = [[[w.text for w in nlp(sents)] for sents in tkns_tnsr[x]] for x in [0,1]]
                    tkns_tnsr = [[s[:min(len(s), opts.maxlen)] + [''] * (opts.maxlen-len(s)) for s in tkns_tnsr[x]] for x in [0,1]]
                    pool_idx = [torch.LongTensor([len(s) - 1 for s in tkns_tnsr[x]]) for x in [0,1]]
                    tkns_tnsr = [batch_to_ids(tkns_tnsr[x]) for x in [0,1]]
                    if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx, weights = [tkns_tnsr[x].to('cuda') for x in [0,1]] , lb_tnsr.to('cuda'), [pool_idx[x].to('cuda') for x in [0,1]], (weights if weights is None else weights.to('cuda'))
                elif task_type == 'nmt':
                    tkns_tnsr, lb_tnsr = [s.split(SC) for s in tkns_tnsr if (type(s) is str and s != '') and len(s) > 0], [list(map(int, s.split(SC))) for s in lb_tnsr if (type(s) is str and s != '') and len(s) > 0]
                    if (len(tkns_tnsr) == 0 or len(lb_tnsr) == 0): continue
                    tkns_tnsr = [s[:min(len(s), opts.maxlen)] + [''] * (opts.maxlen-len(s)) for s in tkns_tnsr]
                    lb_tnsr = torch.LongTensor([s[:min(len(s), opts.maxlen)] + [pad_val] * (opts.maxlen-len(s)) for s in lb_tnsr])
                    pool_idx = torch.LongTensor([len(s) - 1 for s in tkns_tnsr])
                    tkns_tnsr = batch_to_ids(tkns_tnsr)
                    if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx, weights = tkns_tnsr.to('cuda') , lb_tnsr.to('cuda'), pool_idx.to('cuda'), (weights if weights is None else weights.to('cuda'))
                else:
                    tkns_tnsr = [[w.text for w in nlp(text)] for text in tkns_tnsr]
                    if clf.maxpool: tkns_tnsr = [s[:min(len(s), opts.maxlen)] + [''] * (opts.maxlen-len(s)) for s in tkns_tnsr]
                    pool_idx = torch.LongTensor([len(s) - 1 for s in tkns_tnsr])
                    tkns_tnsr = batch_to_ids(tkns_tnsr)
                    if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx, weights = tkns_tnsr.to('cuda') , lb_tnsr.to('cuda'), pool_idx.to('cuda'), (weights if weights is None else weights.to('cuda'))
                # tkns_tnsr = [[[w.text for w in nlp(sents)] for sents in tkns_tnsr[x]] for x in [0,1]] if task_type == 'sentsim' else [[w.text for w in nlp(text)] for text in tkns_tnsr]
                # if (task_type == 'sentsim'): tkns_tnsr = [[s[:min(len(s), opts.maxlen)] + [''] * (opts.maxlen-len(s)) for s in tkns_tnsr[x]] for x in [0,1]]
                # pool_idx = [torch.LongTensor([len(s) - 1 for s in tkns_tnsr[x]]) for x in [0,1]] if task_type == 'sentsim' else torch.LongTensor([len(s) - 1 for s in tkns_tnsr])
                # tkns_tnsr = [batch_to_ids(tkns_tnsr[x]) for x in [0,1]] if task_type == 'sentsim' else batch_to_ids(tkns_tnsr)
                # mask_tnsr = None
                # if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx, mask_tnsr, weights = ([tkns_tnsr[x].to('cuda') for x in [0,1]] if task_type == 'sentsim' else tkns_tnsr.to('cuda')), lb_tnsr.to('cuda'), ([pool_idx[x].to('cuda') for x in [0,1]] if task_type == 'sentsim' else pool_idx.to('cuda')), (mask_tnsr if mask_tnsr is None else mask_tnsr.to('cuda')), (weights if weights is None else weights.to('cuda'))
            else:
                valid_idx = [x for x in range(tkns_tnsr.size(0)) if x not in np.transpose(np.argwhere(tkns_tnsr == -1))[:,0]]
                if (len(valid_idx) == 0): continue
                idx, tkns_tnsr, lb_tnsr, record_idx = [idx[x] for x in range(len(idx)) if x in valid_idx], tkns_tnsr[valid_idx], lb_tnsr[valid_idx], [record_idx[x] for x in range(len(record_idx)) if x in valid_idx] if task_type == 'nmt' else None
                mask_tnsr = tkns_tnsr.eq(pad_val * torch.ones_like(tkns_tnsr)).float()
                mask_tnsr = mask_tnsr if mdl_name == 'trsfmxl' else (mask_tnsr[:, :, 1:].contiguous().view(tkns_tnsr.size(0), -1) if task_type=='sentsim' else mask_tnsr[:, 1:].contiguous())
                if (use_gpu): tkns_tnsr, lb_tnsr, mask_tnsr, weights = tkns_tnsr.to('cuda'), lb_tnsr.to('cuda'), (mask_tnsr if mask_tnsr is None else mask_tnsr.to('cuda')), (weights if weights is None else weights.to('cuda'))
                pool_idx = None if task_type == 'nmt' else tkns_tnsr.eq(clf_tknids[0] * torch.ones_like(tkns_tnsr)).int().argmax(-1)
            clf_loss, lm_loss = clf(input_ids=tkns_tnsr, pool_idx=pool_idx, labels=lb_tnsr.view(-1), weights=weights)
            train_loss = clf_loss.mean() if lm_loss is None else (clf_loss.mean() + lmcoef * ((lm_loss.view(tkns_tnsr.size(0), -1) * mask_tnsr).sum(1) / (1e-12 + mask_tnsr.sum(1))).mean())
            total_loss += train_loss.item()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), clipmaxn)
            train_loss.backward()
            optimizer.step()
        print('Train loss in %i epoch(s): %f' % (epoch + 1, total_loss / (step + 1)))
    save_model(clf, optimizer, '%s_%s.pth' % (task_name, mdl_name), devq=devq)


def eval(clf, dataset, binlbr, clf_tknids, pad_val=0, task_type='mltc-clf', task_name='classification', ds_name='', mdl_name='sota', clipmaxn=0.25, use_gpu=False):
    clf.eval()
    total_loss, indices, preds, probs, all_logits, trues, ds_name = 0, [], [], [], [], [], ds_name.strip()
    if task_type != 'nmt' or task_type != 'entlmnt': dataset.dataset.remove_mostfrqlb()
    for step, batch in enumerate(tqdm(dataset, desc="%s batches" % ds_name.title() if ds_name else 'Evaluation')):
        if task_type == 'nmt':
            idx, tkns_tnsr, lb_tnsr, record_idx = batch
            record_idx = [list(map(int, x.split(SC))) for x in record_idx]
        else:
            idx, tkns_tnsr, lb_tnsr = batch
        indices.extend(idx if type(idx) is list else list(idx))
        _lb_tnsr = lb_tnsr
        if (mdl_name == 'elmo'):
            if task_type == 'entlmnt' or task_type == 'sentsim':
                tkns_tnsr = [[[w.text for w in nlp(sents)] for sents in tkns_tnsr[x]] for x in [0,1]]
                tkns_tnsr = [[s[:min(len(s), opts.maxlen)] + [''] * (opts.maxlen-len(s)) for s in tkns_tnsr[x]] for x in [0,1]]
                pool_idx = _pool_idx = [torch.LongTensor([len(s) - 1 for s in tkns_tnsr[x]]) for x in [0,1]]
                tkns_tnsr = [batch_to_ids(tkns_tnsr[x]) for x in [0,1]]
                if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx= [tkns_tnsr[x].to('cuda') for x in [0,1]] , lb_tnsr.to('cuda'), [pool_idx[x].to('cuda') for x in [0,1]]
            elif task_type == 'nmt':
                tkns_tnsr, lb_tnsr = [s.split(SC) for s in tkns_tnsr if (type(s) is str and s != '') and len(s) > 0], [list(map(int, s.split(SC))) for s in lb_tnsr if (type(s) is str and s != '') and len(s) > 0]
                if (len(tkns_tnsr) == 0 or len(lb_tnsr) == 0): continue
                tkns_tnsr = [s[:min(len(s), opts.maxlen)] + [''] * (opts.maxlen-len(s)) for s in tkns_tnsr]
                _lb_tnsr = lb_tnsr = torch.LongTensor([s[:min(len(s), opts.maxlen)] + [pad_val] * (opts.maxlen-len(s)) for s in lb_tnsr])
                pool_idx = _pool_idx = torch.LongTensor([len(s) - 1 for s in tkns_tnsr])
                tkns_tnsr = batch_to_ids(tkns_tnsr)
                if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx = tkns_tnsr.to('cuda') , lb_tnsr.to('cuda'), pool_idx.to('cuda')
            else:
                tkns_tnsr = [[w.text for w in nlp(text)] for text in tkns_tnsr]
                if clf.maxpool: tkns_tnsr = [s[:min(len(s), opts.maxlen)] + [''] * (opts.maxlen-len(s)) for s in tkns_tnsr]
                pool_idx = _pool_idx = torch.LongTensor([len(s) - 1 for s in tkns_tnsr])
                tkns_tnsr = batch_to_ids(tkns_tnsr)
                if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx= tkns_tnsr.to('cuda') , lb_tnsr.to('cuda'), pool_idx.to('cuda')
        else:
            valid_idx = [x for x in range(tkns_tnsr.size(0)) if x not in np.transpose(np.argwhere(tkns_tnsr == -1))[:,0]]
            if (len(valid_idx) == 0): continue
            _, _, _lb_tnsr, _ = idx, tkns_tnsr, lb_tnsr, record_idx = [idx[x] for x in range(len(idx)) if x in valid_idx], tkns_tnsr[valid_idx], lb_tnsr[valid_idx], ([record_idx[x] for x in range(len(record_idx)) if x in valid_idx] if task_type == 'nmt' else None)
            pool_idx = _pool_idx = tkns_tnsr.eq(clf_tknids[0] * torch.ones_like(tkns_tnsr)).int().argmax(-1)
            if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx = tkns_tnsr.to('cuda'), lb_tnsr.to('cuda'), pool_idx.to('cuda')
        with torch.no_grad():
            logits = clf(tkns_tnsr, pool_idx, labels=None)
            if task_type == 'mltc-clf' or task_type == 'entlmnt':
                loss_func = nn.CrossEntropyLoss(reduction='none')
                loss = loss_func(logits.view(-1, len(binlbr)), lb_tnsr.view(-1))
                prob, pred = torch.softmax(logits, -1).max(-1)
            elif task_type == 'mltl-clf':
                loss_func = nn.MultiLabelSoftMarginLoss(reduction='none')
                loss = loss_func(logits.view(-1, len(binlbr)), lb_tnsr.view(-1, len(binlbr)))
                prob = torch.sigmoid(logits).data
                pred = (prob > opts.pthrshld).int()
            elif task_type == 'nmt':
                loss_func = nn.CrossEntropyLoss(reduction='none')
                loss = loss_func(logits.view(-1, len(binlbr)), lb_tnsr.view(-1))
                prob, pred = torch.softmax(logits, -1).max(-1)
            elif task_type == 'sentsim':
                loss_func = nn.MSELoss(reduction='none')
                loss = loss_func(logits.view(-1), lb_tnsr.view(-1))
                prob, pred = logits, logits
            total_loss += loss.mean().item()
        if task_type == 'nmt':
            last_tkns = torch.arange(_lb_tnsr.size(0)) * _lb_tnsr.size(1) + _pool_idx
            flat_tures, flat_preds, flat_probs = _lb_tnsr.view(-1).tolist(), pred.view(-1).detach().cpu().tolist(), prob.view(-1).detach().cpu().tolist()
            flat_tures_set, flat_preds_set, flat_probs_set = set(flat_tures), set(flat_preds), set(flat_probs)
            trues.append([[max(flat_tures_set, key=flat_tures[a:b][c[idx]:c[idx+1]].count) for idx in range(len(c)-1)] for a, b, c in zip(range(_lb_tnsr.size(0)), last_tkns, record_idx)])
            preds.append([[max(flat_preds_set, key=flat_preds[a:b][c[idx]:c[idx+1]].count) for idx in range(len(c)-1)] for a, b, c in zip(range(_lb_tnsr.size(0)), last_tkns, record_idx)])
            probs.append([[max(flat_probs_set, key=flat_probs[a:b][c[idx]:c[idx+1]].count) for idx in range(len(c)-1)] for a, b, c in zip(range(_lb_tnsr.size(0)), last_tkns, record_idx)])
            # preds.append([flat_preds[a:b] for a, b in zip(range(_lb_tnsr.size(0)), last_tkns)])
            # probs.append([flat_probs[a:b] for a, b in zip(range(_lb_tnsr.size(0)), last_tkns)])
        else:
            trues.append(_lb_tnsr.view(_lb_tnsr.size(0), -1).numpy() if task_type == 'mltl-clf' else _lb_tnsr.view(-1).detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
            probs.append(prob.detach().cpu().numpy())

        all_logits.append(logits.view(_lb_tnsr.size(0), -1, logits.size(-1)).detach().cpu().numpy())
    total_loss = total_loss / (step + 1)
    print('Evaluation loss on %s dataset: %.2f' % (ds_name, total_loss))

    all_logits = np.concatenate(all_logits, axis=0)
    if task_type == 'nmt':
        trues = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(trues))))
        preds = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(preds))))
        probs = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(probs))))
    else:
        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)
        probs = np.concatenate(probs, axis=0)
    resf_prefix = ds_name.lower().replace(' ', '_')
    with open('%s_preds_trues.pkl' % resf_prefix, 'wb') as fd:
        pickle.dump((trues, preds, probs, all_logits), fd)
    if any(task_type == t for t in ['mltc-clf', 'entlmnt', 'nmt']):
        preds = preds
    elif task_type == 'mltl-clf':
        preds = preds
    elif task_type == 'sentsim':
        preds = np.squeeze(preds)
    if task_type == 'sentsim':
        if (np.isnan(preds).any()):
            print('Predictions contain NaN values! Please try to decrease the learning rate!')
            return
        metric_names, metrics_funcs = ['Mean Absolute Error', 'Mean Squared Error', 'Mean Squared Log Error', 'Median Absolute Error', 'R2', 'Pearson Correlation'], [metrics.mean_absolute_error, metrics.mean_squared_error, metrics.mean_squared_log_error, metrics.median_absolute_error, metrics.r2_score, _prsn_cor]
        perf_df = pd.DataFrame(dict([(k, [f(trues, preds)]) for k, f in zip(metric_names, metrics_funcs)]), index=[mdl_name])[metric_names]
    elif task_type == 'mltl-clf':
        perf_df = pd.DataFrame(metrics.classification_report(trues, preds, target_names=[binlbr[x] for x in binlbr.keys()], output_dict=True)).T[['precision', 'recall', 'f1-score', 'support']]
    else:
        perf_df = pd.DataFrame(metrics.classification_report(trues, preds, target_names=[binlbr[x] for x in binlbr.keys() if x in preds or x in trues], output_dict=True)).T[['precision', 'recall', 'f1-score', 'support']]
    print('Results for %s dataset is:\n%s' % (ds_name.title(), perf_df))
    perf_df.to_excel('perf_%s.xlsx' % resf_prefix)
    if (type(indices[0]) is str and SC in indices[0]):
        indices = list(itertools.chain.from_iterable([list(map(int, idx.split(SC))) for idx in indices if idx]))
    try:
        dataset.dataset.fill_labels(preds, saved_path='pred_%s.csv' % resf_prefix, index=indices)
    except Exception as e:
        print(e)


def _prsn_cor(trues, preds):
    return np.corrcoef(trues, preds)[0, 1]


def save_model(model, optimizer, fpath='checkpoint.pth', in_wrapper=False, devq=None, **kwargs):
    if in_wrapper: model = model.module
    model = model.cpu() if devq and len(devq) > 0 else model
    checkpoint = {'model': model, 'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict()}
    checkpoint.update(kwargs)
    torch.save(checkpoint, fpath)


def main():
    if any(opts.task == t for t in ['bc5cdr-chem', 'bc5cdr-dz', 'shareclefe', 'ddi', 'chemprot', 'i2b2', 'hoc', 'mednli', 'biosses', 'clnclsts']):
        main_func = classify
    else:
        return
    if (opts.distrb):
        if (opts.np > 1): # Multi-process multiple GPU
            import torch.multiprocessing as mp
            mp.spawn(main_func, nprocs=len(opts.devq))
        else: # Single-process multiple GPU
            main_func(opts.devq if len(opts.devq) > 1 else opts.devq[0])
    elif (opts.devq): # Single-process
        main_func(opts.devq if len(opts.devq) > 1 else opts.devq[0])
    else:
        main_func(None) # CPU


if __name__ == '__main__':
    # Logging setting
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Parse commandline arguments
    op = OptionParser()
    op.add_option('-k', '--kfold', default=10, action='store', type='int', dest='kfold', help='indicate the K fold cross validation')
    op.add_option('-p', '--pid', default=0, action='store', type='int', dest='pid', help='indicate the process ID')
    op.add_option('-n', '--np', default=1, action='store', type='int', dest='np', help='indicate the number of processes used for training')
    op.add_option('-f', '--fmt', default='npz', help='data stored format: csv, npz, or h5 [default: %default]')
    op.add_option('-s', '--spfmt', default='csr', help='sparse data stored format: csc or csr [default: %default]')
    op.add_option('-a', '--avg', default='micro', help='averaging strategy for performance metrics: micro or macro [default: %default]')
    op.add_option('-j', '--epochs', default=1, action='store', type='int', dest='epochs', help='indicate the epoch used in deep learning')
    op.add_option('-z', '--bsize', default=64, action='store', type='int', dest='bsize', help='indicate the batch size used in deep learning')
    op.add_option('-o', '--omp', action='store_true', dest='omp', default=False, help='use openmp multi-thread')
    op.add_option('-g', '--gpunum', default=1, action='store', type='int', dest='gpunum', help='indicate the gpu device number')
    op.add_option('-q', '--gpuq', dest='gpuq', help='prefered gpu device queue [template: DEVICE_ID1,DEVICE_ID2,...,DEVICE_IDn]')
    op.add_option('--gpumem', default=0.5, action='store', type='float', dest='gpumem', help='indicate the per process gpu memory fraction')
    op.add_option('--crsdev', action='store_true', dest='crsdev', default=False, help='whether to use heterogeneous devices')
    op.add_option('--distrb', action='store_true', dest='distrb', default=False, help='whether to distribute data over multiple devices')
    op.add_option('--distbknd', default='nccl', action='store', dest='distbknd', help='distribute framework backend')
    op.add_option('--disturl', default='env://', action='store', dest='disturl', help='distribute framework url')
    op.add_option('--vocab', dest='vocab', help='vocabulary file')
    op.add_option('--bpe', dest='bpe', help='bpe merge file')
    op.add_option('--maxlen', default=128, action='store', type='int', dest='maxlen', help='indicate the maximum sequence length for each samples')
    op.add_option('--maxtrial', default=50, action='store', type='int', dest='maxtrial', help='maximum time to try')
    op.add_option('--droplast', action='store_true', dest='droplast', default=False, help='whether to drop the last incompleted batch')
    op.add_option('--maxpool', action='store_true', dest='maxpool', default=False, help='whether to use max pooling when selecting features')
    op.add_option('--lr', default=float(1e-3), action='store', type='float', dest='lr', help='indicate the learning rate of the optimizer')
    op.add_option('--wdecay', default=float(1e-5), action='store', type='float', dest='wdecay', help='indicate the weight decay of the optimizer')
    op.add_option('--lmcoef', default=0.5, action='store', type='float', dest='lmcoef', help='indicate the coefficient of the language model loss when fine tuning')
    op.add_option('--pdrop', default=0.2, action='store', type='float', dest='pdrop', help='indicate the dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler')
    op.add_option('--pthrshld', default=0.5, action='store', type='float', dest='pthrshld', help='indicate the threshold for predictive probabilitiy')
    op.add_option('--clipmaxn', default=0.25, action='store', type='float', dest='clipmaxn', help='indicate the max norm of the gradients')
    op.add_option('-i', '--input', help='input dataset')
    op.add_option('-w', '--cache', default='.cache', help='the location of cache files')
    op.add_option('-y', '--year', default='2013', help='the year when the data is released [default: %default]')
    op.add_option('-u', '--task', default='ddi', type='str', dest='task', help='the task name [default: %default]')
    op.add_option('-m', '--model', default='gpt2', type='str', dest='model', help='the model to be validated')
    op.add_option('--pretrained', dest='pretrained', help='pretrained model file')
    op.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False, help='display detailed information')

    (opts, args) = op.parse_args()
    if len(args) > 0:
    	op.print_help()
    	op.error('Please input options instead of arguments.')
    	sys.exit(1)

    # Parse config file
    if (os.path.exists(CONFIG_FILE)):
    	cfgr = io.cfg_reader(CONFIG_FILE)
    	plot_cfg = cfgr('bionlp.util.plot', 'init')
    	plot_common = cfgr('bionlp.util.plot', 'common')
    	# txtclf.init(plot_cfg=plot_cfg, plot_common=plot_common)
    else:
        print('Config file `%s` does not exist!' % CONFIG_FILE)

    if (opts.gpuq is not None and not opts.gpuq.strip().isspace()):
    	opts.gpuq = list(range(torch.cuda.device_count())) if (opts.gpuq == 'auto' or opts.gpuq == 'all') else [int(x) for x in opts.gpuq.split(',') if x]
    elif (opts.gpunum > 0):
        opts.gpuq = list(range(opts.gpunum))
    else:
        opts.gpuq = []
    if (opts.gpuq and opts.gpunum > 0):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, opts.gpuq[:opts.gpunum]))
        setattr(opts, 'devq', list(range(torch.cuda.device_count())))
    else:
        setattr(opts, 'devq', None)

    main()
