#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2021 by Caspar. All rights reserved.
# File Name: shell.py
# Author: Shankai Yan
# E-mail: dr.skyan@gmail.com
# Created Time: 2021-03-29 17:19:30
###########################################################################
#

from torch.utils.data import DataLoader

from ..util.dataset import MaskedLMDataset, MaskedLMIterDataset, BaseIterDataset, ShellDataset, DataParallel
from transformer import BERTClfHead


class BaseShell():
    def __init__(self, model, tokenizer, encode_func, transforms=[], transforms_args={}, transforms_kwargs={}, special_tknids_args=dict(zip(['start_tknids', 'clf_tknids', 'delim_tknids'], ['_@_', ' _$_', ' _#_'])), pad_val=0):
        self.model = model
        self.tokenizer = tokenizer
        self.encode_func = encode_func
        self.transforms = transforms
        self.transforms_args = transforms_args
        self.transforms_kwargs = transforms_kwargs
        self.special_tknids_args = special_tknids_args
        self.pad_val = pad_val

    def __call__(self, *inputs, **kwargs):
        use_gpu = next(self.model.parameters()).is_cuda
        batch_size, n_jobs, droplast = kwargs.setdefault('bsize', 16), kwargs.setdefault('n_jobs', 1), kwargs.setdefault('droplast', False)
        ds = ShellDataset(inputs[0], encode_func=self.encode_func, tokenizer=self.tokenizer, transforms=self.transforms, transforms_args=self.transforms_args, transforms_kwargs=self.transforms_kwargs)
        if isinstance(self.model, BERTClfHead) or type(self.model) is DataParallel and isinstance(self.model.module, BERTClfHead): ds = MaskedLMIterDataset(ds) if isinstance(ds, BaseIterDataset) else MaskedLMDataset(ds)
        ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, sampler=None, num_workers=n_jobs, drop_last=droplast)
        clf_tknids = self.special_tknids_args['clf_tknids']
        clf_h = []
        for step, batch in enumerate(tqdm(ds_loader, desc='[Totally %i samples] Retrieving predictions/embeddings' % len(ds))):
            idx, tkns_tnsr, lb_tnsr = batch[:3]
            extra_inputs = batch[3:]
            mask_tnsr = (~tkns_tnsr.eq(self.pad_val * torch.ones_like(tkns_tnsr))).long()
            pool_idx = tkns_tnsr.eq(clf_tknids[0] * torch.ones_like(tkns_tnsr)).int().argmax(-1)
            if use_gpu: tkns_tnsr, pool_idx, mask_tnsr, extra_inputs = tkns_tnsr.to('cuda'), pool_idx.to('cuda'), mask_tnsr.to('cuda'), tuple([x.to('cuda') for x in extra_inputs])
            h = self.model(tkns_tnsr, mask_tnsr if isinstance(self.model, BERTClfHead) or type(self.model) is DataParallel and isinstance(self.model.module, BERTClfHead) else pool_idx, *extra_inputs, embedding_mode=kwargs.setdefault('embedding_mode', False))
            clf_h.append(h.detach().cpu().view(tkns_tnsr.size(0), -1))
            del tkns_tnsr, mask_tnsr, pool_idx, h
        return torch.cat(clf_h, dim=0)

    def encode(self, text):
        if type(text) is list:
            return [self._transform_chain((self.encode_func(x, self.tokenizer), None), self.transforms, self.transforms_args, self.transforms_kwargs)[0] for x in text]
        else:
            return self._transform_chain((self.encode_func(text, self.tokenizer), None), self.transforms, self.transforms_args, self.transforms_kwargs)[0]

    def decode(self, logits):
        if not hasattr(self.model, 'binlbr'): return logits
        if type(text) is list:
            return [self.model.binlbr[x] for x in logits]
        else:
            return self.model.binlbr[logits]

    def _transform_chain(self, sample, transforms, transforms_args={}, transforms_kwargs={}):
        if transforms:
            transforms = transforms if type(transforms) is list else [transforms]
            transforms_kwargs = transforms_kwargs if type(transforms_kwargs) is list else [transforms_kwargs]
            for transform, transform_kwargs in zip(transforms, transforms_kwargs):
                transform_kwargs.update(transforms_args)
                sample = transform(sample, **transform_kwargs) if callable(transform) else getattr(self.model, transform)(sample, **transform_kwargs)
        return sample

