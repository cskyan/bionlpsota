#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2021 by Caspar. All rights reserved.
# File Name: dataset.py
# Author: Shankai Yan
# E-mail: dr.skyan@gmail.com
# Created Time: 2021-03-29 17:34:10
###########################################################################
#

import os, sys, random, operator, pickle, itertools, logging
from collections import OrderedDict

import numpy as np
import pandas as pd

import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset
from torch.nn.parallel import replicate

from bionlp.util import io

global FALSE_LABELS
FALSE_LABELS = [0, '0', 'false', 'False', 'F']


class DatasetInterface(object):
    def __init__(self):
        raise NotImplementedError

    def _transform_chain(self, sample):
        if self.transforms:
            self.transforms = self.transforms if type(self.transforms) is list else [self.transforms]
            self.transforms_kwargs = self.transforms_kwargs if type(self.transforms_kwargs) is list else [self.transforms_kwargs]
            for transform, transform_kwargs in zip(self.transforms, self.transforms_kwargs):
                transform_kwargs.update(self.transforms_args)
                sample = transform(sample, **transform_kwargs) if callable(transform) else getattr(self, transform)(sample, **transform_kwargs)
        return sample

    def _nmt_transform(self, sample):
        return sample[0], [self.binlb.setdefault(y, len(self.binlb)) for y in sample[1]]

    def _mltl_nmt_transform(self, sample, get_lb=None):
        get_lb = (lambda x: x.split(self.sc)) if get_lb is None else get_lb
        labels = [get_lb(lb) for lb in sample[1]]
        return sample[0], [[self.binlb.setdefault(y, len(self.binlb)) for y in lbs] if type(lbs) is list else self.binlb.setdefault(lbs, len(self.binlb)) for lbs in labels]

    def _binc_transform(self, sample):
        return sample[0], 1 if sample[1] in self.binlb else 0

    def _mltc_transform(self, sample):
        return sample[0], self.binlb.setdefault(sample[1], len(self.binlb))

    def _mltl_transform(self, sample, get_lb=None):
        get_lb = (lambda x: x.split(self.sc)) if get_lb is None else get_lb
        labels = get_lb(sample[1])
        return sample[0], [1 if lb in labels else 0 for lb in self.binlb.keys()]

    def fill_labels(self, lbs, binlb=True, index=None, saved_col='preds', saved_path=None, **kwargs):
        if binlb and self.binlbr is not None:
            lbs = [(';'.join([self.binlbr[l] for l in np.where(lb == 1)[0]]) if self.mltl else ','.join(['_'.join([str(i), str(l)]) for i, l in enumerate(lb)])) if hasattr(lb, '__iter__') else (self.binlbr[lb] if len(self.binlbr) > 1 else (next(x for x in self.binlbr.values()) if lb==1 else '')) for lb in lbs]
        filled_df = self._df.copy(deep=True)[~self._df.index.duplicated(keep='first')]
        try:
            if index is not None:
                filled_df.loc[index, saved_col] = lbs
            else:
                filled_df[saved_col] = lbs
        except Exception as e:
            logging.warning(e)
            with open('pred_lbs.tmp', 'wb') as fd:
                pickle.dump((filled_df, index, self.label_col, lbs), fd)
            raise e
        if (saved_path is not None):
            filled_df.to_csv(saved_path, sep='\t', **kwargs)
        return filled_df

    def rebalance(self):
        if (self.binlb is None): return
        task_cols, task_trsfm = self.config.task_col, self.config.task_trsfm
        lb_trsfm = [x['get_lb'] for x in task_trsfm[1] if 'get_lb' in x]
        self.df = self._df
        if len(lb_trsfm) > 0:
            lb_df = self.df[task_cols['y']].apply(lb_trsfm[0])
        else:
            lb_df = self.df[task_cols['y']]
        if (type(lb_df.iloc[0]) is list):
            lb_df[:] = [self._mltl_transform((None, self.sc.join(lbs)))[1] for lbs in lb_df]
            max_lb_df = lb_df.loc[[idx for idx, lbs in lb_df.iteritems() if np.sum(list(map(int, lbs))) == 0]]
            max_num, avg_num = max_lb_df.shape[0], 1.0 * lb_df[~lb_df.index.isin(max_lb_df.index)].shape[0] / len(lb_df.iloc[0])
        else:
            class_count = np.array([[1 if lb in y else 0 for lb in self.binlb.keys()] for y in lb_df if y is not None]).sum(axis=0)
            max_num, max_lb_bin = class_count.max(), class_count.argmax()
            max_lb_df = lb_df[lb_df == self.binlbr[max_lb_bin]]
            avg_num = np.mean([class_count[x] for x in range(len(class_count)) if x != max_lb_bin])
        removed_idx = max_lb_df.sample(n=int(max_num-avg_num), random_state=1).index
        self.df = self.df.loc[list(set(self.df.index)-set(removed_idx))]

    def remove_mostfrqlb(self):
        if (self.binlb is None or self.binlb == 'rgrsn'): return
        task_cols, task_trsfm = self.config.task_col, self.config.task_trsfm
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


class BaseDataset(DatasetInterface, Dataset):
    """Basic dataset class"""

    def __init__(self, csv_file, tokenizer, config, sampw=False, sampfrac=None, **kwargs):
        self.sc = config.sc
        self.config = config
        self.text_col = [str(s) for s in config.task_col['X']] if hasattr(config.task_col['X'], '__iter__') and type(config.task_col['X']) is not str else str(config.task_col['X'])
        self.label_col = [str(s) for s in config.task_col['y']] if hasattr(config.task_col['y'], '__iter__') and type(config.task_col['y']) is not str else str(config.task_col['y'])
        self.df = self._df = csv_file if type(csv_file) is pd.DataFrame else pd.read_csv(csv_file, sep=config.dfsep, engine='python', error_bad_lines=False, index_col=config.task_col['index'])
        self.df[config.task_col['y']] = ['false' if lb is None or lb is np.nan or (type(lb) is str and lb.isspace()) else lb for lb in self.df[config.task_col['y']]] # convert all absent labels to negative
        logging.info('Input DataFrame size: %s' % str(self.df.shape))
        if sampfrac: self.df = self._df = self._df.sample(frac=float(sampfrac))
        self.df.columns = self.df.columns.astype(str, copy=False)
        self.mltl = config.task_ext_params.setdefault('mltl', False)
        self.sample_weights = sampw

        # Construct the binary label mapping
        binlb = (config.task_ext_params['binlb'] if 'binlb' in config.task_ext_params else None) if 'binlb' not in kwargs else kwargs['binlb']
        if (binlb == 'rgrsn'): # regression tasks
            self.df[self.label_col] = self.df[self.label_col].astype('float')
            self.binlb = None
            self.binlbr = None
            self.df = self.df[self.df[self.label_col].notnull()]
        elif (type(binlb) is str and binlb.startswith('mltl')): # multi-label classification tasks
            sc = binlb.split(self.sc)[-1]
            self.df = self.df[self.df[self.label_col].notnull()]
            lb_df = self.df[self.label_col]
            labels = sorted(set([lb for lbs in lb_df for lb in lbs.split(sc)])) if type(lb_df.iloc[0]) is not list else sorted(set([lb for lbs in lb_df for lb in lbs]))
            self.binlb = OrderedDict([(lb, i) for i, lb in enumerate(labels)])
            self.mltl = True
        elif (binlb is None): # normal cases
            lb_df = self.df[self.df[self.label_col].notnull()][self.label_col]
            labels = sorted(set(lb_df)) if type(lb_df.iloc[0]) is not list else sorted(set([lb for lbs in lb_df for lb in lbs]))
            if len(labels) == 1: labels = ['false'] + labels
            self.binlb = OrderedDict([(lb, i) for i, lb in enumerate(labels)])
        else: # previously constructed
            self.binlb = binlb
        if self.binlb: self.binlbr = OrderedDict([(i, lb) for lb, i in self.binlb.items()])
        self.encode_func = config.encode_func
        self.tknz_kwargs = config.tknz_kwargs
        self.tokenizer = tokenizer
        if hasattr(tokenizer, 'vocab'):
            self.vocab_size = len(tokenizer.vocab)
        elif hasattr(tokenizer, 'vocab_size'):
            self.vocab_size = tokenizer.vocab_size
        # Combine all the data transformers
        self.transforms = config.task_trsfm[0] + config.mdl_trsfm[0]
        self.transforms_kwargs = config.task_trsfm[1] + config.mdl_trsfm[1]
        self.transforms_args = kwargs.setdefault('transforms_args', {}) # Common transformer kwargs
        config.register_callback('mdl_trsfm', BaseDataset.callback_update_trsfm(self))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = self.encode_func(record[self.text_col], self.tokenizer, self.tknz_kwargs), record[self.label_col]
        sample = self._transform_chain(sample)
        return (self.df.index[idx], torch.tensor(sample[0][0]) if type(sample[0][0]) is not str else sample[0], torch.tensor(sample[1])) + tuple([torch.tensor(x) for x in sample[0][1:]] if type(sample[0][0]) is not str else [torch.tensor(x) if x[0] is not str else x[0] for x in sample[0][1:]]) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ())

    @classmethod
    def callback_update_trsfm(cls, dataset):
        def _callback(config):
            dataset.transforms = config.task_trsfm[0] + config.mdl_trsfm[0]
            dataset.transforms_kwargs = config.task_trsfm[1] + config.mdl_trsfm[1]
        return _callback

class _Extext(object):
    def __init__(self, extext_path, config):
        if os.path.isdir(extext_path):
            self.extext_path = extext_path
        elif extext_path.startswith('http://') and 'solr' in extext_path:
            import pysolr
            self.solr_url, self.solr_txt_field = self.extext_path.split(config.sc)
            self.solr = pysolr.Solr(self.solr_url, results_cls=dict)
        else:
            self.extext_path = '.'

    def get_txt(self, identifier, ret_list=False):
        if hasattr(self, 'extext_path'):
            ext_fpath = os.path.join(self.extext_path, identifier)
            with open(ext_fpath, 'r') as fd:
                extext = fd.readlines() if ret_list else ' '.join(fd.readlines())
        elif hasattr(self, 'solr'):
            res = self.solr.search('id:%s' % identifier)['response']
            if res['numFound'] == 0:
                extext = [] if ret_list else ''
            else:
                extext = res['docs'][0][self.solr_txt_field] if ret_list else ' '.join(res['docs'][0][self.solr_txt_field])
        return extext


class ExtextBaseDataset(BaseDataset):
    def __init__(self, csv_file, tokenizer, config, sampw=False, sampfrac=None, extext_path='.', **kwargs):
        super(ExtextBaseDataset, self).__init__(csv_file, text_col, label_col, encode_func, tokenizer, config, sep=sep, binlb=binlb, transforms=transforms, transforms_args=transforms_args, transforms_kwargs=transforms_kwargs, mltl=mltl, sampw=sampw, sampfrac=sampfrac, **kwargs)
        self.extext = _Extext(extext_path)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = self.encode_func(self.extext.get_txt(record[self.text_col], ret_list=False), self.tokenizer, self.tknz_kwargs), record[self.label_col]
        sample = self._transform_chain(sample)
        return (self.df.index[idx], torch.tensor(sample[0][0]) if type(sample[0][0]) is not str else sample[0], torch.tensor(sample[1])) + tuple([torch.tensor(x) for x in sample[0][1:]] if type(sample[0][0]) is not str else []) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ())


class SentSimDataset(BaseDataset):
    """Sentence Similarity task dataset class"""

    def __init__(self, csv_file, tokenizer, config, sampw=False, sampfrac=None, **kwargs):
        super(SentSimDataset, self).__init__(csv_file, text_col, label_col, encode_func, tokenizer, config, sep=sep, skip_blank_lines=False, keep_default_na=False, na_values=[], binlb=binlb, transforms=transforms, transforms_args=transforms_args, transforms_kwargs=transforms_kwargs, sampw=sampw, sampfrac=sampfrac, **kwargs)
        self.ynormfunc, self.ynormfuncr = config.ynormfunc if hasattr(config, 'ynormfunc') else (None, None)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = [self.encode_func(record[sent_idx], self.tokenizer, self.tknz_kwargs) for sent_idx in self.text_col], record[self.label_col]
        sample = self._transform_chain(sample)
        return (self.df.index[idx], torch.tensor(sample[0][0]), torch.tensor(0. if sample[1] is np.nan else (float(sample[1]) if self.ynormfunc is None else self.ynormfunc(float(sample[1]))))) + tuple([torch.tensor(x) for x in sample[0][1:]]) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ())

    def fill_labels(self, lbs, index=None, saved_path=None, **kwargs):
        lbs = lbs if self.ynormfuncr is None else list(map(self.ynormfuncr, lbs))
        filled_df = self._df.copy(deep=True)[~self._df.index.duplicated(keep='first')]
        if index:
            filled_df.loc[index, self.label_col] = lbs
        else:
            filled_df[self.label_col] = lbs
        if (saved_path is not None):
            filled_df.to_csv(saved_path, **kwargs)
        return filled_df


class EntlmntDataset(BaseDataset):
    """Entailment task dataset class"""

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = self.encode_func([record[sent_idx] for sent_idx in self.text_col], self.tokenizer, self.tknz_kwargs), record[self.label_col] if self.label_col in record and record[self.label_col] is not np.nan else [k for k in FALSE_LABELS if k in self.binlb][0]
        sample = self._transform_chain(sample)
        return (self.df.index[idx], torch.tensor(sample[0][0]), torch.tensor(sample[1])) + tuple([torch.tensor(x) for x in sample[0][1:]]) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ())


class NERDataset(BaseDataset):
    """NER task dataset class"""

    def __init__(self, csv_file, text_col, label_col, encode_func, tokenizer, config, sep='\t', binlb=None, transforms=[], transforms_args={}, transforms_kwargs=[], sampw=False, sampfrac=None, lb_coding='IOB', **kwargs):
        super(NERDataset, self).__init__(csv_file, text_col, label_col, encode_func, tokenizer, config, sep=sep, header=None if csv_file.endswith('tsv') else 0, skip_blank_lines=False, keep_default_na=False, na_values=[], binlb=binlb, transforms=transforms, transforms_args=transforms_args, transforms_kwargs=transforms_kwargs, sampw=sampw, sampfrac=sampfrac, **kwargs)
        all_binlb = set([k.split('-')[-1] for k in self.binlb.keys() if k != 'O'])
        encoded_lbs = ['%s-%s' % (pre, lb) for lb in all_binlb for pre in set(lb_coding) - set(['O']) if lb and not lb.isspace()] + ['O']
        self.binlb = OrderedDict([(lb, i) for i, lb in enumerate(encoded_lbs)])
        self.binlbr = OrderedDict([(i, lb) for i, lb in enumerate(encoded_lbs)])
        self.num_lbs = len(encoded_lbs)
        sep_selector = self.df[self.text_col].apply(lambda x: True if x=='.' else False)
        sep_selector.iloc[-1] = False if sep_selector.iloc[-2] else True
        int_idx = pd.DataFrame(np.arange(self.df.shape[0]), index=self.df.index)
        self.boundaries = [0] + list(itertools.chain.from_iterable((int_idx[sep_selector.values].values+1).tolist()))

    def __len__(self):
        return len(self.boundaries) - 1

    def __getitem__(self, idx):
        record = self.df.iloc[self.boundaries[idx]:self.boundaries[idx+1]].dropna()
        sample = self.encode_func(record[self.text_col].values.tolist(), self.tokenizer, self.tknz_kwargs), record[self.label_col].values.tolist()
        sample = list(map(list, zip(*[(x, y) for x, y in zip(*sample) if x and y])))
        num_samples = [len(x) for x in sample[0]] if (len(sample[0]) > 0 and type(sample[0][0]) is list) else [1] * len(sample[0])
        record_idx = [0] + np.cumsum(num_samples).tolist()
        is_empty = (type(sample[0]) is list and len(sample[0]) == 0) or (type(sample[0]) is list and len(sample[0]) > 0 and all([type(x) is list and len(x) == 0 for x in sample[0]]))
        if (is_empty): return self.sc.join(map(str, record.index.values.tolist())), '' if self.encode_func == _tokenize else torch.LongTensor([-1]*self.config.maxlen), '' if self.encode_func == _tokenize else torch.LongTensor([-1]*self.config.maxlen), self.sc.join(map(str, record_idx))
        is_encoded = (type(sample[0]) is list and type(sample[0][0]) is int) or (type(sample[0]) is list and len(sample[0]) > 0 and type(sample[0][0]) is list and len(sample[0][0]) > 0 and type(sample[0][0][0]) is int)
        sample = list(itertools.chain.from_iterable(sample[0])) if is_encoded else sample[0], list(itertools.chain.from_iterable([[x] * ns for x, ns in zip(sample[1], num_samples)]))
        sample = self._transform_chain(sample)
        return (self.sc.join(map(str, record.index.values.tolist())), (torch.tensor(sample[0][0]) if is_encoded else self.sc.join(sample[0][0])), (torch.tensor(sample[1]) if is_encoded else self.sc.join(map(str, sample[1]))), self.sc.join(map(str, record_idx))) + tuple([torch.tensor(x) for x in sample[0][1:]]) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ())

    def fill_labels(self, lbs, saved_path=None, binlb=True, index=None, **kwargs):
        if binlb and self.binlbr is not None:
            lbs = [self.binlbr[lb] for lb in lbs]
        filled_df = self._df.copy(deep=True)[~self._df.index.duplicated(keep='first')]
        if index:
            filled_df[self.label_col] = ''
            filled_df.loc[index, self.label_col] = lbs
        else:
            filled_df[self.label_col] = lbs
        if (saved_path is not None):
            filled_df.to_csv(saved_path, sep='\t', header=None, index=None, **kwargs)
        return filled_df


class MaskedLMDataset(BaseDataset):
    """Wrapper dataset class for masked language model"""

    def __init__(self, dataset, special_tknids=[101, 102, 102, 103], masked_lm_prob=0.15):
        self._ds = dataset
        self.config = self._ds.config
        self.text_col = self._ds.text_col if hasattr(self._ds, 'text_col') else None
        self.label_col = self._ds.label_col if hasattr(self._ds, 'label_col') else None
        self.mltl = self._ds.mltl if hasattr(self._ds, 'mltl') else None
        self.binlb = self._ds.binlb if hasattr(self._ds, 'binlb') else None
        self.binlbr = self._ds.binlbr if hasattr(self._ds, 'binlbr') else None
        self.encode_func = self._ds.encode_func
        self.tokenizer = self._ds.tokenizer
        self.vocab_size = self._ds.vocab_size
        self.transforms = self._ds.transforms
        self.transforms_args = self._ds.transforms_args
        self.transforms_kwargs = self._ds.transforms_kwargs
        self.special_tknids = special_tknids
        self.masked_lm_prob = masked_lm_prob
        if hasattr(self._ds, 'df'): self.df = self._ds.df
        if hasattr(self._ds, '_df'): self._df = self._ds._df

    def __len__(self):
        return self._ds.__len__()

    def __getitem__(self, idx):
        orig_sample = self._ds[idx]
        sample = orig_sample[1], orig_sample[2]
        masked_lm_ids = np.array(sample[0])
        pad_trnsfm_idx = self.transforms.index(_pad_transform) if len(self.transforms) > 0 and _pad_transform in self.transforms else -1
        pad_trnsfm_kwargs = self.transforms_kwargs[pad_trnsfm_idx] if pad_trnsfm_idx and pad_trnsfm_idx in self.transforms_kwargs > -1 else {}
        if type(self._ds) in [EntlmntDataset, SentSimDataset] and len(self.transforms_kwargs) >= 2 and self.transforms_kwargs[1].setdefault('sentsim_func', None) is not None:
            masked_lm_lbs = [np.array([-1 if x in self.special_tknids + [pad_trnsfm_kwargs.setdefault('xpad_val', -1)] else x for x in sample[0][X]]) for X in [0,1]]
            valid_idx = [np.where(masked_lm_lbs[x] > -1)[0] for x in [0,1]]
            cand_samp_idx = [random.sample(range(len(valid_idx[x])), min(self.config.maxlen, max(1, int(round(len(valid_idx[x]) * self.masked_lm_prob))))) for x in [0,1]]
            cand_idx = [valid_idx[x][cand_samp_idx[x]] for x in [0,1]]
            rndm = [np.random.uniform(low=0, high=1, size=(len(cand_idx[x]),)) for x in [0,1]]
            for x in [0,1]:
                masked_lm_ids[x][cand_idx[x][rndm[x] < 0.8]] = self.special_tknids[-1]
                masked_lm_ids[x][cand_idx[x][rndm[x] >= 0.9]] = random.randrange(0, self.vocab_size)
            for X in [0,1]:
                masked_lm_lbs[X][list(filter(lambda x: x not in cand_idx[X], range(len(masked_lm_lbs[X]))))] = -1
        else:
            masked_lm_lbs = np.array([-1 if x in self.special_tknids + [pad_trnsfm_kwargs.setdefault('xpad_val', -1)] else x for x in sample[0]])
            valid_idx = np.where(masked_lm_lbs > -1)[0]
            cand_samp_idx = random.sample(range(len(valid_idx)), min(self.config.maxlen, max(1, int(round(len(valid_idx) * self.masked_lm_prob)))))
            cand_idx = valid_idx[cand_samp_idx]
            rndm = np.random.uniform(low=0, high=1, size=(len(cand_idx),))
            masked_lm_ids[cand_idx[rndm < 0.8]] = self.special_tknids[-1]
            masked_lm_ids[cand_idx[rndm >= 0.9]] = random.randrange(0, self.vocab_size)
            masked_lm_lbs[list(filter(lambda x: x not in cand_idx, range(len(masked_lm_lbs))))] = -1
        segment_ids = torch.zeros(masked_lm_ids.shape)
        if type(self._ds) in [EntlmntDataset, SentSimDataset] and (len(self.transforms_kwargs) < 2 or self.transforms_kwargs[1].setdefault('sentsim_func', None) is None):
            segment_idx = sample[0].eq(self.special_tknids[2] * torch.ones_like(sample[0])).int()
            segment_idx = torch.where(segment_idx)
            segment_start, segment_end = torch.tensor([segment_idx[-1][i] for i in range(0, len(segment_idx[-1]), 2)]), torch.tensor([segment_idx[-1][i] for i in range(1, len(segment_idx[-1]), 2)])
            idx = torch.arange(sample[0].size(-1)) * torch.ones_like(sample[0])
            segment_ids = ((idx > segment_start.view(-1 if len(idx.size()) == 1 else (idx.size()[0], 1)) * torch.ones_like(idx)) & (idx <= segment_end.view(-1 if len(idx.size()) == 1 else (idx.size()[0], 1)) * torch.ones_like(idx))).int()
        return orig_sample + (torch.tensor(masked_lm_ids), torch.tensor(masked_lm_lbs), torch.tensor(segment_ids).long())
        # return self.df.index[idx], (sample[0] if type(sample[0]) is str or type(sample[0][0]) is str else torch.tensor(sample[0])), torch.tensor(sample[1]), torch.tensor(masked_lm_ids), torch.tensor(masked_lm_lbs)

    def fill_labels(self, lbs, index=None, saved_path=None, **kwargs):
        return self._ds.fill_labels(lbs, index=index, saved_path=saved_path, **kwargs)


class ConceptREDataset(BaseDataset):
    """Relation extraction task with pre-annotated concepts dataset class"""

    def __init__(self, csv_file, text_col, label_col, encode_func, tokenizer, config, cw2v_model=None, sep='\t', binlb=None, transforms=[], transforms_args={}, transforms_kwargs=[], sampw=False, sampfrac=None, **kwargs):
        super(ConceptREDataset, self).__init__(csv_file, text_col, label_col, encode_func, tokenizer, config, sep=sep, binlb=binlb, transforms=transforms, transforms_args=transforms_args, transforms_kwargs=transforms_kwargs, sampw=sampw, sampfrac=sampfrac, **kwargs)
        self.cw2v_model = cw2v_model

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = self.encode_func(record[self.text_col], self.tokenizer, self.tknz_kwargs), record[self.label_col]
        sample = self._transform_chain(sample)
        cncpt_ids = [self.cw2v_model.vocab[record[col]].index for col in ['cid1', 'cid2']] if self.cw2v_model else []
        return (self.df.index[idx], torch.tensor(sample[0][0]), torch.tensor(sample[1]), torch.tensor(cncpt_ids)) + tuple([torch.tensor(x) for x in sample[0][1:]]) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ())


class ShellDataset(BaseDataset):
    """Dataset class for Shell objects that handle free text data"""

    def __init__(self, text, encode_func, tokenizer, transforms=[], transforms_args={}, transforms_kwargs=[], **kwargs):
        self.text = [text] if type(text) is str else list(text)
        self.encode_func = encode_func
        self.tokenizer = tokenizer
        if hasattr(tokenizer, 'vocab'):
            self.vocab_size = len(tokenizer.vocab)
        elif hasattr(tokenizer, 'vocab_size'):
            self.vocab_size = tokenizer.vocab_size
        self.transforms = transforms
        self.transforms_args = transforms_args
        self.transforms_kwargs = transforms_kwargs

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        sample = self.encode_func(text, self.tokenizer, self.tknz_kwargs), None
        sample = self._transform_chain(sample)
        return (idx, torch.tensor(sample[0][0]), torch.tensor(sample[1])) + tuple([torch.tensor(x) for x in sample[0][1:]])


class EmbeddingDataset(BaseDataset):
    def __init__(self, embeddings, labels=None):
        self.embeddings = embeddings
        self.labels = labels
        if self.labels is not None: assert(len(self.embeddings)==len(self.labels))

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), self.labels[idx] if self.labels else None


class EmbeddingPairDataset(BaseDataset):
    def __init__(self, embeddings1, embeddings2, pair_indices, labels=None):
        self.embeddings1 = torch.tensor(embeddings1)
        self.embeddings2 = torch.tensor(embeddings2)
        self.pair_indices = pair_indices
        self.labels = labels
        indices1, indices2 = zip(*pair_indices)
        assert(max(indices1) < len(embeddings1) and max(indices2) < len(embeddings2))
        if self.labels is not None: assert(len(self.pair_indices)==len(self.labels))

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        indices = self.pair_indices[idx]
        sample = torch.stack((self.embeddings1[indices[0]], self.embeddings2[indices[1]])).view(1, 2, -1)
        return (sample, self.labels[idx]) if self.labels else sample


class OntoDataset(BaseDataset):
    def __init__(self, csv_file, text_col, label_col, encode_func, tokenizer, config, onto_col='ontoid', sep='\t', index_col='id', binlb=None, transforms=[], transforms_args={}, transforms_kwargs=[], sampw=False, sampfrac=None, **kwargs):
        super(OntoDataset, self).__init__(csv_file, text_col, label_col, encode_func, tokenizer, config, sep=sep, index_col=index_col, binlb=binlb, transforms=transforms, transforms_args=transforms_args, transforms_kwargs=transforms_kwargs, sampw=sampw, sampfrac=sampfrac, **kwargs)
        if hasattr(config, 'onto_df') and type(config.onto_df) is pd.DataFrame:
            self.onto = config.onto_df
        else:
            onto_fpath = config.onto if hasattr(config, 'onto') and os.path.exists(config.onto) else 'onto.csv'
            logging.info('Reading ontology dictionary file [%s]...' % onto_fpath)
            self.onto = pd.read_csv(onto_fpath, sep=sep, index_col=index_col)
            setattr(config, 'onto_df', self.onto)
        logging.info('Ontology DataFrame size: %s' % str(self.onto.shape))
        self.onto2id = dict([(k, i+1) for i, k in enumerate(self.onto.index)])
        self.onto_col = onto_col

    def __getitem__(self, idx):
        record = self.df.iloc[idx]
        sample = self.encode_func([record[sent_idx] for sent_idx in self.text_col], self.tokenizer, self.tknz_kwargs), record[self.label_col] if self.label_col in record and record[self.label_col] is not np.nan else [k for k in [0, '0', 'false', 'False', 'F'] if k in self.binlb][0]
        sample = self._transform_chain(sample)
        return (self.df.index[idx], torch.tensor(sample[0][0]), torch.tensor(sample[1])) + tuple([torch.tensor(x) for x in sample[0][1:]]) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ()) + (torch.tensor(self.onto2id[record[self.onto_col]]),)


class BaseIterDataset(DatasetInterface, IterableDataset):
    """ Basic iterable dataset """

    def __init__(self, fpath, text_col, label_col, encode_func, tokenizer, config, sep='\t', index_col='id', binlb=None, transforms=[], transforms_args={}, transforms_kwargs=[], mltl=False, sampw=False, **kwargs):
        self.sc = self.config.sc
        self.config = config
        self.index_col = index_col
        self.text_col = [str(s) for s in text_col] if hasattr(text_col, '__iter__') and type(text_col) is not str else str(text_col)
        self.label_col = [str(s) for s in label_col] if hasattr(label_col, '__iter__') and type(label_col) is not str else str(label_col)
        self.mltl = mltl
        self.sample_weights = sampw
        self.data = io.JsonIterable(fpath, retgen=True) if os.path.splitext(fpath)[-1] == '.json' else io.DataFrameIterable(fpath, retgen=True, sep=sep)
        self._df = pd.read_csv(fpath, sep=sep, dtype={self.index_col:object}).set_index(self.index_col) if type(self.data) is io.DataFrameIterable else None
        self.encode_func = encode_func
        self.tokenizer = tokenizer
        if hasattr(tokenizer, 'vocab'):
            self.vocab_size = len(tokenizer.vocab)
        elif hasattr(tokenizer, 'vocab_size'):
            self.vocab_size = tokenizer.vocab_size
        self.transforms = transforms
        self.transforms_args = transforms_args
        self.transforms_kwargs = transforms_kwargs

        self.data, preproc_data = itertools.tee(self.data)
        labels = set([])
        if (binlb == 'rgrsn'):
            self.binlb = None
            self.binlbr = None
        elif (type(binlb) is str and binlb.startswith('mltl')):
            sc = binlb.split(self.sc)[-1]
            for record in preproc_data:
                labels |= set(record[self.label_col].split(sc))
            self.binlb = OrderedDict([(lb, i) for i, lb in enumerate(sorted(labels))])
            self.mltl = True
        elif (binlb is None):
            for record in preproc_data:
                labels.add(record[self.label_col])
            self.binlb = OrderedDict([(lb, i) for i, lb in enumerate(sorted(labels))])
        else:
            self.binlb = binlb
        if self.binlb: self.binlbr = OrderedDict([(i, lb) for lb, i in self.binlb.items()])

    def _transform(self, record):
        sample = self.encode_func(record[self.text_col], self.tokenizer, self.tknz_kwargs), record[self.label_col]
        sample = self._transform_chain(sample)
        return (record[self.index_col], (sample[0] if type(sample[0]) is str or type(sample[0][0]) is str else torch.tensor(sample[0])), torch.tensor(sample[1])) + tuple([torch.tensor(x) for x in sample[0][1:]]) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ())

    def __iter__(self):
        return iter((self._transform(record) for record in self.data))

    def fill_labels(self, lbs, binlb=True, index=None, saved_col='preds', saved_path=None, **kwargs):
        if self._df is None: return
        return DatasetInterface.fill_labels(self, lbs, binlb=binlb, index=index, saved_col=saved_col, saved_path=saved_path, **kwargs)


class EntlmntIterDataset(BaseIterDataset):
    """Entailment task iterable dataset class"""

    def _transform(self, record):
        sample = [self.encode_func(' '.join([record[sent_idx] for sent_idx in self.text_col[:-1]]), self.tokenizer, self.tknz_kwargs), self.encode_func(record[self.text_col[-1]], self.tokenizer, self.tknz_kwargs)], record[self.label_col]
        sample = self._transform_chain(sample)
        return (record[self.index_col], torch.tensor(sample[0][0]), torch.tensor(sample[1])) + tuple([torch.tensor(x) for x in sample[0][1:]]) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ())


class EntlmntMltlIterDataset(EntlmntIterDataset):
    """Entailment task transformed from multi-label classification iterable dataset class"""

    def __init__(self, fpath, text_col, label_col, encode_func, tokenizer, config, sep='\t', index_col='id', binlb=None, transforms=[], transforms_args={}, transforms_kwargs=[], sampw=False, sent_mode=False, **kwargs):
        try:
            super(EntlmntMltlIterDataset, self).__init__(fpath, text_col, label_col, encode_func, tokenizer, config, sep=sep, index_col=index_col, binlb=binlb, transforms=transforms, transforms_args=transforms_args, transforms_kwargs=transforms_kwargs, mltl=False, sampw=sampw, **kwargs)
        except Exception as e:
            logging.warning(e)
        if type(binlb) is str and binlb.startswith('mltl'):
            self.data, preproc_data = itertools.tee(self.data)
            labels = set([])
            sc = binlb.split(self.sc)[-1]
            for record in preproc_data:
                labels |= set(record[kwargs['origlb']] if type(record[kwargs['origlb']]) is list else (record[kwargs['origlb']].split(sc) if type(record[kwargs['origlb']]) is str else []))
            self.orig_binlb = OrderedDict([(lb, i) for i, lb in enumerate(sorted(labels))])
            self.orig_binlbr = OrderedDict([(i, lb) for lb, i in self.orig_binlb.items()])
            self.mltl = True
        self.sent_mode = sent_mode
        if sent_mode: self.data = EntlmntMltlIterDataset._split_to_sents(self.data, index_col=index_col, text_col=text_col[0], lb_col=kwargs['origlb'], lb_sep=sc if 'sc' in locals() else ';', keep_locs=kwargs.setdefault('keep_locs', False), update_locs=kwargs.setdefault('update_locs', False))
        self.data = EntlmntMltlIterDataset._mltl2entlmnt(self.data, text_col, kwargs['origlb'], label_col, local_lb_col=kwargs.setdefault('locallb', 'local_label'), ref_lb_col=kwargs.setdefault('reflb', None), binlb=binlb, lbtxt=kwargs.setdefault('lbtxt', None), sampw=sampw, neglbs=kwargs.setdefault('neglbs', None))
        self.origlb = kwargs['origlb']
        self.locallb = kwargs['locallb']
        self.binlb_str = binlb
        self.binlb = OrderedDict([('false', 0), ('include', 1)])
        self.binlbr = OrderedDict([(0, 'false'), (1, 'include')])

    @classmethod
    def _mltl2entlmnt(cls, data, text_col, orig_lb_col, dest_lb_col, local_lb_col='local_label', ref_lb_col=None, binlb=None, lbtxt=None, sampw=False, neglbs=None, record_sc=';;'):
        sc = binlb.split(record_sc)[-1]
        for record in data:
            ref_lbs = set(record[ref_lb_col] if type(record[ref_lb_col]) is list else (record[ref_lb_col].split(sc) if type(record[ref_lb_col]) is str else [])) if ref_lb_col is not None and ref_lb_col in record else None
            for lb in sorted(filter(lambda x: x, set(record[orig_lb_col] if type(record[orig_lb_col]) is list else (record[orig_lb_col].split(sc) if type(record[orig_lb_col]) is str else [])))):
                record[local_lb_col] = lb
                record[dest_lb_col] = 'include' if ref_lbs is None or len(ref_lbs) == 0 or lb in ref_lbs else 'false'
                if sampw:
                    notes, nwghts = ([lb],[1.0]) if lbtxt is None else lbtxt['func'](lb, sampw=True, **lbtxt['kwargs'])
                else:
                    notes = [lb] if lbtxt is None else lbtxt['func'](lb, **lbtxt['kwargs'])
                    nwghts = [1.0] * len(notes) if type(notes) is list else 1.0
                notes = [notes] if type(notes) is not list else notes
                nwghts = [nwghts] if type(nwghts) is not list else nwghts
                for note, w in zip(notes, nwghts):
                    record[text_col[-1]] = note
                    if sampw: record['weight'] = w
                    yield record
            if neglbs is not None:
                if sampw: record['weight'] = 1.0
                for neg_lb in neglbs['func'](record[orig_lb_col], **neglbs['kwargs']):
                    record[local_lb_col] = neg_lb
                    record[dest_lb_col] = ''
                    notes = [neg_lb] if lbtxt is None else lbtxt['func'](neg_lb, **lbtxt['kwargs'])
                    for note in notes:
                        record[text_col[-1]] = note
                        yield record

    @classmethod
    def _split_to_sents(cls, data, index_col='id', text_col='text', lb_col='labels', lb_sep=';', keep_locs=False, update_locs=False):
        from bionlp.util import func
        splitted_data = []
        for record in data:
            doc_id = record[index_col]
            orig_labels = record[lb_col].split(lb_sep) if record[lb_col] is not None and type(record[lb_col]) is str and not record[lb_col].isspace() else []
            if len(orig_labels) == 0:
                yield record
                continue
            labels, locs = zip(*[lb.split('|') for lb in orig_labels])
            doc = nlp(record[text_col])
            sent_bndrs = [(s.start_char, s.end_char) for s in doc.sents]
            lb_locs = [tuple(map(int, loc.split(':'))) for loc in locs]
            if (np.amax(lb_locs) > np.amax(sent_bndrs)): lb_locs = np.array(lb_locs) - np.amin(lb_locs) # Temporary fix
            # overlaps = list(filter(lambda x: len(x) > 0, [func.overlap_tuple(lb_locs, sb, ret_idx=True) for sb in sent_bndrs]))
            overlaps = [func.overlap_tuple(lb_locs, sb, ret_idx=True) for sb in sent_bndrs]
            overlaps = [ovr if len(ovr)>0 else ((),()) for ovr in overlaps]
            indices, locss = zip(*overlaps) if len(overlaps) > 0 else ([[]]*len(sent_bndrs),[[]]*len(sent_bndrs))
            indices, locss = list(map(list, indices)), list(map(list, locss))
            miss_aligned = list(set(range(len(labels))) - set(func.flatten_list(indices)))
            if len(miss_aligned) > 0:
                for i in range(len(sent_bndrs)):
                    indices[i].extend(miss_aligned)
                    locss[i].extend([sent_bndrs[i]]*len(miss_aligned))
            last_idx = 0
            for i, (idx, locs, sent) in enumerate(zip(indices, locss, doc.sents)):
                lbs = [labels[x] for x in idx]
                record[index_col] = '_'.join(map(str, [doc_id, i]))
                record[text_col] = sent.text
                record[lb_col] = lb_sep.join(['|'.join([lb, ':'.join(map(str, np.array(loc)-(last_idx if update_locs else 0)))]) for lb, loc in zip(lbs, locs)] if keep_locs else lbs)
                last_idx += sent.end_char
                yield record

    def fill_labels(self, lbs, binlb=True, index=None, saved_col='preds', saved_path=None, **kwargs):
        if self._df is None: return
        sc = self.binlb_str.split(self.sc)[-1]
        num_underline =  list(str(self._df.index[0])).count('_')
        index = ['_'.join(str(idx).split('_')[:num_underline+1]) for idx in index]
        combined_df = pd.DataFrame({'id':index, 'lbs':lbs}).groupby('id').apply(lambda x: sc.join(map(str, x['lbs'].values)))
        index, lbs = combined_df.index, combined_df.values
        lbs = [sc.join([origlb for origlb, lb in zip(origlbs.split(sc), lbstr.split(sc)) if lb == '1']) if type(origlbs) is str and type(lbstr) is str else '' for origlbs, lbstr in zip(self._df[self.origlb], lbs)]
        binlb=False
        return BaseIterDataset.fill_labels(self, lbs, binlb=binlb, index=index, saved_col=saved_col, saved_path=saved_path, **kwargs)


class OntoIterDataset(EntlmntMltlIterDataset):
    def __init__(self, fpath, text_col, label_col, encode_func, tokenizer, config, onto_fpath='onto.csv', onto_col='ontoid', sep='\t', index_col='id', binlb=None, transforms=[], transforms_args={}, transforms_kwargs=[], sampw=False, sent_mode=False, **kwargs):
        def _get_onto_lb(lb, sampw=False, onto_df=None):
            if onto_df is not None and lb in onto_df.index: return (onto_df.loc[lb]['label'], 1) if sampw else onto_df.loc[lb]['label']
        def _get_onto_def(lb, sampw=False, onto_df=None, onto_dict_df=None):
            if onto_dict_df is not None and lb in onto_dict_df.index:
                # return ((list(set(onto_dict_df.loc[lb]['text'].tolist())) if type(onto_dict_df.loc[lb]['text']) is pd.Series else [onto_dict_df.loc[lb]['text']]) if onto_dict_df.loc[lb]['text'] is not np.nan else [t for t in onto_dict_df.loc[lb][['label','exact_synm','relate_synm','narrow_synm','broad_synm']] if t is not np.nan][:2]) if lb in onto_dict_df.index else []
                if lb not in onto_dict_df.index:
                    return [[],[]] if sampw else []
                elif onto_dict_df.loc[lb]['text'] is not np.nan:
                    notes = list(set(onto_dict_df.loc[lb]['text'].tolist())) if type(onto_dict_df.loc[lb]['text']) is pd.Series else [onto_dict_df.loc[lb]['text']]
                    return (notes, [1.0]*len(notes)) if sampw else notes
                else:
                    notes = [t for t in onto_dict_df.loc[lb][['label','exact_synm','relate_synm','narrow_synm','broad_synm']] if t is not np.nan][:2]
                    return (notes, [0.5]*len(notes)) if sampw else notes
            elif onto_df is not None and lb in onto_df.index:
                return (onto_df.loc[lb]['label'], 0.5) if sampw else onto_df.loc[lb]['label']
        onto_fpaths = onto_fpath.split(self.sc) if onto_fpath and type(onto_fpath) is str else None
        onto_fpath, onto_dict_fpath = (onto_fpaths[0], onto_fpaths[1] if len(onto_fpaths) > 1 else None) if onto_fpath is not None else (None, None)
        if onto_fpath and os.path.exists(onto_fpath):
            self.onto = pd.read_csv(onto_fpath, sep=sep, index_col=index_col)
            self.onto2id = dict([(k, i+1) for i, k in enumerate(self.onto.index)])
        if onto_dict_fpath and os.path.exists(onto_dict_fpath):
            self.onto_dict = pd.read_csv(onto_dict_fpath, sep=sep, index_col=index_col)
        self.onto_col = onto_col
        kwargs['lbtxt'] = dict(func=_get_onto_def, kwargs={'onto_df':self.onto, 'onto_dict_df':self.onto_dict}) if text_col[-1] != 'onto' and hasattr(self, 'onto_dict') else dict(func=_get_onto_lb, kwargs={'onto_df':self.onto})
        super(OntoIterDataset, self).__init__(fpath, text_col, label_col, encode_func, tokenizer, config, sep=sep, index_col=index_col, binlb=binlb, transforms=transforms, transforms_args=transforms_args, transforms_kwargs=transforms_kwargs, sampw=sampw, sent_mode=sent_mode, **kwargs)

    def _transform(self, record):
        sample = [self.encode_func(' '.join([record[sent_idx] for sent_idx in self.text_col[:-1]]), self.tokenizer, self.tknz_kwargs), self.encode_func(record[self.text_col[-1]], self.tokenizer, self.tknz_kwargs)], record[self.label_col]
        sample = self._transform_chain(sample)
        return (record[self.index_col], torch.tensor(sample[0][0]), torch.tensor(sample[1])) + tuple([torch.tensor(x) for x in sample[0][1:]]) + ((torch.tensor(record['weight'] if 'weight' in record else 1.0),) if self.sample_weights else ()) + (torch.tensor(self.onto2id[record[self.onto_col]] if hasattr(self, 'onto_col') and self.onto_col in record else 0),)

    def fill_labels(self, lbs, index=None, saved_path=None, **kwargs):
        return EntlmntMltlIterDataset.fill_labels(self, lbs, index=index, saved_col='rerank_preds', saved_path=saved_path, **kwargs)


class OntoExtextIterDataset(OntoIterDataset):
    def __init__(self, fpath, text_col, label_col, encode_func, tokenizer, config, onto_fpath='onto.csv', onto_col='ontoid', sep='\t', index_col='id', binlb=None, transforms=[], transforms_args={}, transforms_kwargs=[], sampw=False, sent_mode=False, extext_path='.', **kwargs):
        super(OntoExtextIterDataset, self).__init__(fpath, text_col, label_col, encode_func, tokenizer, config, onto_fpath=onto_fpath, onto_col=onto_col, sep=sep, index_col=index_col, binlb=binlb, transforms=transforms, transforms_args=transforms_args, transforms_kwargs=transforms_kwargs, sampw=sampw, sent_mode=sent_mode, **kwargs)
        self.extext = _Extext(extext_path)
        self.data = self._extext(data)

    def _extext(self, data, lb_sep=';', keep_locs=False):
        for record in data:
            orig_labels = record[self.label_col].split(lb_sep) if record[self.label_col] is not None and type(record[self.label_col]) is str and not record[lb_col].isspace() else []
            loc_map = {}
            if len(orig_labels) > 0:
                labels, locs = zip(*[lb.split('|') for lb in orig_labels])
                segids, locs = zip(*[loc.split('-') for loc in locs])
                segids, parids = zip(*list(map(lambda x: x if len(x) > 1 else [0, x[0]], [seg.split('>') for seg in segids])))
                for seg, par, lb, loc in zip(segids, parids, labels, locs):
                    loc_map.setdefault(seg, {}).setdefault(par, []).append((lb, loc))
            for i, sent_idx in enumerate(self.text_col[:-1]):
                extexts = self.extext.get_txt(record[sent_idx], ret_list=True)
                for j, extext in enumerate(extexts):
                    record[sent_idx] = extext
                    # Adjust labels
                    labels = loc_map.setdefault(i, {}).setdefault(j, [])
                    record[self.label_col] = lb_sep.join(map(operator.itemgetter(0), labels) if not keep_locs else ['|'.join(x) for x in labels])
                    yield record


class MaskedLMIterDataset(BaseIterDataset):
    """Wrapper iterable dataset class for masked language model"""

    def __init__(self, dataset, config, special_tknids=[101, 102, 102, 103], masked_lm_prob=0.15):
        self.config = config
        self._ds = dataset
        self.index_col = self._ds.index_col
        self.text_col = self._ds.text_col
        self.label_col = self._ds.label_col
        self.mltl = self._ds.mltl
        self.binlb = self._ds.binlb
        self.binlbr = self._ds.binlbr
        self.encode_func = self._ds.encode_func
        self.tokenizer = self._ds.tokenizer
        self.vocab_size = self._ds.vocab_size
        self.transforms = self._ds.transforms
        self.transforms_args = self._ds.transforms_args
        self.transforms_kwargs = self._ds.transforms_kwargs
        self._transform_chain = self._ds._transform_chain
        self.special_tknids = special_tknids
        self.masked_lm_prob = masked_lm_prob
        self.data = dataset.data

    def _transform(self, record):
        orig_sample = self._ds._transform(record)
        sample = orig_sample[1], orig_sample[2]
        masked_lm_ids = np.array(sample[0])
        pad_trnsfm_idx = self.transforms.index(_pad_transform) if len(self.transforms) > 0 and _pad_transform in self.transforms else -1
        pad_trnsfm_kwargs = self.transforms_kwargs[pad_trnsfm_idx] if pad_trnsfm_idx and pad_trnsfm_idx in self.transforms_kwargs > -1 else {}
        if isinstance(self._ds, EntlmntIterDataset) and len(self.transforms_kwargs) >= 2 and self.transforms_kwargs[1].setdefault('sentsim_func', None) is not None:
            masked_lm_lbs = [np.array([-1 if x in self.special_tknids + [pad_trnsfm_kwargs.setdefault('xpad_val', -1)] else x for x in sample[0][X]]) for X in [0,1]]
            valid_idx = [np.where(masked_lm_lbs[x] > -1)[0] for x in [0,1]]
            cand_samp_idx = [random.sample(range(len(valid_idx[x])), min(self.config.maxlen, max(1, int(round(len(valid_idx[x]) * self.masked_lm_prob))))) for x in [0,1]]
            cand_idx = [valid_idx[x][cand_samp_idx[x]] for x in [0,1]]
            rndm = [np.random.uniform(low=0, high=1, size=(len(cand_idx[x]),)) for x in [0,1]]
            for x in [0,1]:
                masked_lm_ids[x][cand_idx[x][rndm[x] < 0.8]] = self.special_tknids[-1]
                masked_lm_ids[x][cand_idx[x][rndm[x] >= 0.9]] = random.randrange(0, self.vocab_size)
            for X in [0,1]:
                masked_lm_lbs[X][list(filter(lambda x: x not in cand_idx[X], range(len(masked_lm_lbs[X]))))] = -1
        else:
            masked_lm_lbs = np.array([-1 if x in self.special_tknids + [pad_trnsfm_kwargs.setdefault('xpad_val', -1)] else x for x in sample[0]])
            valid_idx = np.where(masked_lm_lbs > -1)[0]
            cand_samp_idx = random.sample(range(len(valid_idx)), min(self.config.maxlen, max(1, int(round(len(valid_idx) * self.masked_lm_prob)))))
            cand_idx = valid_idx[cand_samp_idx]
            rndm = np.random.uniform(low=0, high=1, size=(len(cand_idx),))
            masked_lm_ids[cand_idx[rndm < 0.8]] = self.special_tknids[-1]
            masked_lm_ids[cand_idx[rndm >= 0.9]] = random.randrange(0, self.vocab_size)
            masked_lm_lbs[list(filter(lambda x: x not in cand_idx, range(len(masked_lm_lbs))))] = -1
        segment_ids = torch.zeros(masked_lm_ids.shape)
        if isinstance(self._ds, EntlmntIterDataset) and (len(self.transforms_kwargs) < 2 or self.transforms_kwargs[1].setdefault('sentsim_func', None) is None):
            segment_idx = sample[0].eq(self.special_tknids[2] * torch.ones_like(sample[0])).int()
            segment_idx = torch.where(segment_idx)
            segment_start, segment_end = torch.tensor([segment_idx[-1][i] for i in range(0, len(segment_idx[-1]), 2)]), torch.tensor([segment_idx[-1][i] for i in range(1, len(segment_idx[-1]), 2)])
            idx = torch.arange(sample[0].size(-1)) * torch.ones_like(sample[0])
            segment_ids = ((idx > segment_start.view(-1 if len(idx.size()) == 1 else (idx.size()[0], 1)) * torch.ones_like(idx)) & (idx <= segment_end.view(-1 if len(idx.size()) == 1 else (idx.size()[0], 1)) * torch.ones_like(idx))).int()
        return orig_sample + (torch.tensor(masked_lm_ids), torch.tensor(masked_lm_lbs), torch.tensor(segment_ids).long())

    def __iter__(self):
        return iter((self._transform(record) for record in self.data))

    def fill_labels(self, lbs, index=None, saved_path=None, **kwargs):
        return self._ds.fill_labels(lbs, index=index, saved_path=saved_path, **kwargs)


class DataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def replicate(self, module, device_ids):
        replicas = super().replicate(module, device_ids)
        for attr in dir(module):
            attr_obj = getattr(module, attr)
            if type(attr_obj) is list and all([isinstance(x, nn.Module) for x in attr_obj]):
                rplcs = zip(*[replicate(x, device_ids, not torch.is_grad_enabled()) for x in attr_obj])
                for mdl, rplc in zip(replicas, rplcs):
                    setattr(mdl, attr, rplc)
            elif isinstance(attr_obj, nn.Module):
                for sub_attr in dir(attr_obj):
                    sub_attr_obj = getattr(attr_obj, sub_attr)
                    if type(sub_attr_obj) is list and all([isinstance(x, nn.Module) for x in sub_attr_obj]):
                        rplcs = zip(*[replicate(x, device_ids, not torch.is_grad_enabled()) for x in sub_attr_obj])
                        for mdl, rplc in zip(replicas, rplcs):
                            setattr(getattr(mdl, attr), sub_attr, rplc)
        return replicas
