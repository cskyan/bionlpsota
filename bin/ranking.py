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

import os, sys, ast, time, copy, random, operator, pickle, string, logging, itertools
from collections import OrderedDict
from optparse import OptionParser
from tqdm import tqdm

import numpy as np
from scipy.sparse import csc_matrix
import pandas as pd

import torch
from torch.utils.data.sampler import WeightedRandomSampler

from sklearn import metrics

from transformers import get_linear_schedule_with_warmup

from bionlp.nlp import enrich_txt_by_w2v
from bionlp.util import io, system

from ..util.dataset import EmbeddingPairDataset
from ..modules.embedding import EmbeddingHead

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
SC=';;'
NUM_TRIM = 0

opts, args = {}, []
cfgr = None


def siamese_rank(dev_id=None):
    '''Predict candidate labels from all available ones using pre-trained model and rank the predictions using siamese network, assuming that each label has a gold standard sample'''
    print('### Siamese Rank Mode ###')
    orig_task = opts.task
    opts.task = opts.task + '_siamese'
    # Prepare model related meta data
    mdl_name = opts.model.split('_')[0].lower().replace(' ', '_')
    common_cfg = cfgr('validate', 'common')
    pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
    config_kwargs = dict([(k, v) for k, v in opts.__dict__.items() if not k.startswith('_') and k not in set(['dataset', 'model', 'template']) and v is not None and type(v) is not function])
    config = Configurable(opts.task, mdl_name, common_cfg=common_cfg, wsdir=PAR_DIR, sc=SC, **config_kwargs)
    params = pr('LM', config.lm_params) if mdl_name != 'none' else {}
    use_gpu = dev_id is not None
    encode_func = config.encode_func
    tokenizer = config.tknzr.from_pretrained(params['pretrained_vocab_path'] if 'pretrained_vocab_path' in params else config.lm_mdl_name) if config.tknzr else None
    task_type = config.task_type
    spcl_tkns = config.lm_tknz_extra_char if config.lm_tknz_extra_char else ['_@_', ' _$_', ' _#_']
    special_tkns = (['start_tknids', 'clf_tknids', 'delim_tknids'], spcl_tkns[:3]) if task_type in ['entlmnt', 'sentsim'] else (['start_tknids', 'clf_tknids'], spcl_tkns[:2])
    special_tknids = _adjust_encoder(mdl_name, tokenizer, config, special_tkns[1], ret_list=True)
    special_tknids_args = dict(zip(special_tkns[0], special_tknids))
    task_trsfm_kwargs = dict(list(zip(special_tkns[0], special_tknids))+[('model',opts.model), ('sentsim_func', opts.sentsim_func), ('seqlen',opts.maxlen)])
    # Prepare task related meta data
    task_path, task_dstype, task_cols, task_trsfm, task_extrsfm, task_extparms = opts.input if opts.input and os.path.isdir(os.path.join(DATA_PATH, opts.input)) else config.task_path, config.task_ds, config.task_col, config.task_trsfm, config.task_ext_trsfm, config.task_ext_params
    trsfms = ([] if hasattr(config, 'embed_type') and config.embed_type else task_extrsfm[0]) + (task_trsfm[0] if len(task_trsfm) > 0 else [])
    trsfms_kwargs = ([] if hasattr(config, 'embed_type') and config.embed_type else ([{'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}] if config.task_type=='nmt' else [{'seqlen':opts.maxlen, 'trimlbs':task_extparms.setdefault('trimlbs', False), 'required_special_tkns':['start_tknids', 'clf_tknids', 'delim_tknids'] if task_type in ['entlmnt', 'sentsim'] and (task_extparms.setdefault('sentsim_func', None) is None or not mdl_name.startswith('bert')) else ['start_tknids', 'clf_tknids'], 'special_tkns':special_tknids_args}, task_trsfm_kwargs, {'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}])) + (task_trsfm[1] if len(task_trsfm) >= 2 else [{}] * len(task_trsfm[0]))
    ds_kwargs = {'sampw':opts.sample_weights, 'sampfrac':opts.sampfrac}
    ds_kwargs.update({'ynormfunc':task_extparms.setdefault('ynormfunc', None)})
    ext_params = dict([(k, getattr(opts, k)) if hasattr(opts, k) else (k, v) for k, v in config.clf_ext_params.items()])
    if hasattr(config, 'embed_type') and config.embed_type: ext_params['embed_type'] = config.embed_type
    task_params = dict([(k, getattr(opts, k)) if hasattr(opts, k) and getattr(opts, k) is not None else (k, v) for k, v in task_extparms.setdefault('mdlcfg', {}).items()])
    print('Classifier hyper-parameters: %s' % ext_params)
    print('Classifier task-related parameters: %s' % task_params)

    if (opts.resume):
        # Load model
        clf, prv_optimizer, resume, chckpnt = load_model(opts.resume)
        if opts.refresh:
            print('Refreshing and saving the model with newest code...')
            try:
                save_model(clf, prv_optimizer, '%s_%s.pth' % (opts.task, opts.model))
            except Exception as e:
                print(e)
        prv_task_params = copy.deepcopy(clf.task_params)
        # Update parameters
        clf.update_params(task_params=task_params, **ext_params)
        clf.to_siamese()
        if (use_gpu): clf = _handle_model(clf, dev_id=dev_id, distrb=opts.distrb)
        optmzr_cls = config.optmzr if config.optmzr else (torch.optim.Adam, {}, None)
        optimizer = optmzr_cls[0](clf.parameters(), lr=opts.lr, weight_decay=opts.wdecay, **optmzr_cls[1]) if opts.optim == 'adam' else torch.optim.SGD(clf.parameters(), lr=opts.lr, momentum=0.9)
        if len(resume) > 0 and prv_optimizer: optimizer.load_state_dict(prv_optimizer.state_dict())
        training_steps = int(len(train_ds) / opts.bsize) if hasattr(train_ds, '__len__') else opts.trainsteps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opts.wrmprop, num_training_steps=training_steps) if not opts.noschdlr and len(optmzr_cls) > 2 and optmzr_cls[2] and optmzr_cls[2] == 'linwarm' else None
        print((optimizer, scheduler))

    else:
        # Build model
        lm_model = gen_mdl(mdl_name, pretrained=True if type(opts.pretrained) is str and opts.pretrained.lower() == 'true' else opts.pretrained, use_gpu=use_gpu, distrb=opts.distrb, dev_id=dev_id) if mdl_name != 'none' else None

        clf = gen_clf(opts.model, opts.encoder, lm_model=lm_model, constraints=opts.cnstrnts.split(',') if opts.cnstrnts else [], task_type=task_type, mlt_trnsfmr=True if task_type in ['entlmnt', 'sentsim'] and task_params.setdefault('sentsim_func', None) is not None else False, task_params=task_params, use_gpu=use_gpu, distrb=opts.distrb, dev_id=dev_id, **ext_params)
        clf.to_siamese()
        optmzr_cls = config.optmzr if config.optmzr else (torch.optim.Adam, {}, None)
        optimizer = optmzr_cls[0](clf.parameters(), lr=opts.lr, weight_decay=opts.wdecay, **optmzr_cls[1]) if opts.optim == 'adam' else torch.optim.SGD(clf.parameters(), lr=opts.lr, momentum=0.9)
        training_steps = int(len(train_ds) / opts.bsize) if hasattr(train_ds, '__len__') else opts.trainsteps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opts.wrmprop, num_training_steps=training_steps) if not opts.noschdlr and len(optmzr_cls) > 2 and optmzr_cls[2] and optmzr_cls[2] == 'linwarm' else None
        print((optimizer, scheduler))

    # Prepare data
    print('Dataset path: %s' % os.path.join(DATA_PATH, task_path))
    train_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'train_siamese.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms else None, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): train_ds = MaskedLMIterDataset(train_ds) if isinstance(train_ds, BaseIterDataset) else MaskedLMDataset(train_ds)
    lb_trsfm = [x['get_lb'] for x in task_trsfm[1] if 'get_lb' in x]
    if (not opts.weight_class or task_type == 'sentsim'):
        class_count = None
    elif len(lb_trsfm) > 0:
        lb_df = train_ds.df[task_cols['y']].apply(lb_trsfm[0])
        class_count = np.array([[1 if lb in y else 0 for lb in train_ds.binlb.keys()] for y in lb_df]).sum(axis=0)
    else:
        lb_df = train_ds.df[task_cols['y']]
        binlb = task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else train_ds.binlb
        class_count = lb_df.value_counts()[binlb.keys()].values
    if (class_count is None):
        class_weights = None
        sampler = None
    else:
        class_weights = torch.Tensor(1.0 / class_count)
        class_weights /= class_weights.sum()
        sampler = WeightedRandomSampler(weights=class_weights, num_samples=opts.bsize, replacement=True)
        if type(dev_id) is list: class_weights = class_weights.repeat(len(dev_id))
    train_loader = DataLoader(train_ds, batch_size=opts.bsize, shuffle=False, sampler=None, num_workers=opts.np, drop_last=opts.droplast)

    dev_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'dev_siamese.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else train_ds.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): dev_ds = MaskedLMIterDataset(dev_ds) if isinstance(dev_ds, BaseIterDataset) else MaskedLMDataset(dev_ds)
    dev_loader = DataLoader(dev_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)
    test_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'test_siamese.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else train_ds.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): test_ds = MaskedLMIterDataset(test_ds) if isinstance(test_ds, BaseIterDataset) else MaskedLMDataset(test_ds)
    test_loader = DataLoader(test_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)

    # Training on doc/sent-pair datasets
    train(clf, optimizer, train_loader, special_tknids_args, scheduler=scheduler, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), weights=class_weights, lmcoef=opts.lmcoef, clipmaxn=opts.clipmaxn, epochs=opts.epochs, earlystop=opts.earlystop, earlystop_delta=opts.es_delta, earlystop_patience=opts.es_patience, task_type=task_type, task_name=opts.task, mdl_name=opts.model, use_gpu=use_gpu, devq=dev_id, resume=resume if opts.resume else {})

    # Evaluating on the doc/sent-pair dev and test sets
    eval(clf, dev_loader, dev_ds.binlbr, special_tknids_args, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='dev_siamese', mdl_name=opts.model, use_gpu=use_gpu, ignored_label=task_extparms.setdefault('ignored_label', None))
    eval(clf, test_loader, test_ds.binlbr, special_tknids_args, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='test_siamese', mdl_name=opts.model, use_gpu=use_gpu, ignored_label=task_extparms.setdefault('ignored_label', None))

    # Adjust the model
    clf.merge_siamese(tokenizer=tokenizer, encode_func=config.encode_func, trnsfm=[trsfms, {}, trsfms_kwargs], special_tknids_args=special_tknids_args, pad_val=task_extparms.setdefault('xpad_val', 0), topk=128, lbnotes='../lbnotes.csv')

    # Recover the original task
    opts.task = orig_task
    clf.task_params = prv_task_params
    # Prepare model related meta data
    task_type = config.task_type
    # Prepare task related meta data
    task_path, task_dstype, task_cols, task_trsfm, task_extrsfm, task_extparms = opts.input if opts.input and os.path.isdir(os.path.join(DATA_PATH, opts.input)) else config.task_path, config.task_ds, config.task_col, config.task_trsfm, config.task_ext_trsfm, config.task_ext_params
    trsfms = ([] if hasattr(config, 'embed_type') and config.embed_type else task_extrsfm[0]) + (task_trsfm[0] if len(task_trsfm) > 0 else [])
    trsfms_kwargs = ([] if hasattr(config, 'embed_type') and config.embed_type else ([{'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}] if config.task_type=='nmt' else [{'seqlen':opts.maxlen, 'trimlbs':task_extparms.setdefault('trimlbs', False), 'required_special_tkns':['start_tknids', 'clf_tknids', 'delim_tknids'] if task_type in ['entlmnt', 'sentsim'] and (task_extparms.setdefault('sentsim_func', None) is None or not mdl_name.startswith('bert')) else ['start_tknids', 'clf_tknids'], 'special_tkns':special_tknids_args}, task_trsfm_kwargs, {'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}])) + (task_trsfm[1] if len(task_trsfm) >= 2 else [{}] * len(task_trsfm[0]))
    # Prepare dev and test sets
    del ds_kwargs['ynormfunc']
    dev_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'dev.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else clf.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): dev_ds = MaskedLMIterDataset(train_ds) if isinstance(train_ds, BaseIterDataset) else MaskedLMDataset(dev_ds)
    dev_loader = DataLoader(dev_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)
    test_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'test.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else clf.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): test_ds = MaskedLMIterDataset(train_ds) if isinstance(train_ds, BaseIterDataset) else MaskedLMDataset(test_ds)
    test_loader = DataLoader(test_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)
    # Evaluation
    eval(clf, dev_loader, dev_ds.binlbr, special_tknids_args, pad_val=task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='dev', mdl_name=opts.model, use_gpu=use_gpu, ignored_label=task_extparms.setdefault('ignored_label', None))
    if opts.traindev: train(clf, optimizer, dev_loader, special_tknids_args, scheduler=scheduler, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), weights=class_weights, lmcoef=opts.lmcoef, clipmaxn=opts.clipmaxn, epochs=opts.epochs, earlystop=opts.earlystop, earlystop_delta=opts.es_delta, earlystop_patience=opts.es_patience, task_type=task_type, task_name=opts.task, mdl_name=opts.model, use_gpu=use_gpu, devq=dev_id)
    eval(clf, test_loader, test_ds.binlbr, special_tknids_args, pad_val=task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='test', mdl_name=opts.model, use_gpu=use_gpu, ignored_label=task_extparms.setdefault('ignored_label', None))


def simsearch_smsrerank(dev_id=None):
    print('### Search Siamese Re-rank Mode ###')
    orig_task = opts.task
    opts.task = opts.task + '_simsearch'
    # Prepare model related meta data
    mdl_name = opts.model.split('_')[0].lower().replace(' ', '_')
    common_cfg = cfgr('validate', 'common')
    pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
    config_kwargs = dict([(k, v) for k, v in opts.__dict__.items() if not k.startswith('_') and k not in set(['dataset', 'model', 'template']) and v is not None and type(v) is not function])
    config = Configurable(opts.task, mdl_name, common_cfg=common_cfg, wsdir=PAR_DIR, sc=SC, **config_kwargs)
    params = pr('LM', config.lm_params) if mdl_name != 'none' else {}
    use_gpu = dev_id is not None
    encode_func = config.encode_func
    tokenizer = config.tknzr.from_pretrained(params['pretrained_vocab_path'] if 'pretrained_vocab_path' in params else config.lm_mdl_name) if config.tknzr else None
    task_type = 'sentsim' #TASK_TYPE_MAP[opts.task]
    spcl_tkns = config.lm_tknz_extra_char if config.lm_tknz_extra_char else ['_@_', ' _$_', ' _#_']
    special_tkns = (['start_tknids', 'clf_tknids', 'delim_tknids'], spcl_tkns[:3]) if task_type in ['entlmnt', 'sentsim'] else (['start_tknids', 'clf_tknids'], spcl_tkns[:2])
    special_tknids = _adjust_encoder(mdl_name, tokenizer, config, special_tkns[1], ret_list=True)
    special_tknids_args = dict(zip(special_tkns[0], special_tknids))
    task_trsfm_kwargs = dict(list(zip(special_tkns[0], special_tknids))+[('model',opts.model), ('sentsim_func', opts.sentsim_func), ('seqlen',opts.maxlen)])
    # Prepare task related meta data
    task_path, task_dstype, task_cols, task_trsfm, task_extrsfm, task_extparms = opts.input if opts.input and os.path.isdir(os.path.join(DATA_PATH, opts.input)) else config.task_path, config.task_ds, config.task_col, config.task_trsfm, config.task_ext_trsfm, config.task_ext_params
    trsfms = ([] if hasattr(config, 'embed_type') and config.embed_type else task_extrsfm[0]) + (task_trsfm[0] if len(task_trsfm) > 0 else [])
    trsfms_kwargs = ([] if hasattr(config, 'embed_type') and config.embed_type else ([{'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}] if config.task_type=='nmt' else [{'seqlen':opts.maxlen, 'trimlbs':task_extparms.setdefault('trimlbs', False), 'required_special_tkns':['start_tknids', 'clf_tknids', 'delim_tknids'] if task_type in ['entlmnt', 'sentsim'] and (task_extparms.setdefault('sentsim_func', None) is None or not mdl_name.startswith('bert')) else ['start_tknids', 'clf_tknids'], 'special_tkns':special_tknids_args}, task_trsfm_kwargs, {'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}])) + (task_trsfm[1] if len(task_trsfm) >= 2 else [{}] * len(task_trsfm[0]))
    ds_kwargs = {'sampw':opts.sample_weights, 'sampfrac':opts.sampfrac}
    ds_kwargs.update({'ynormfunc':task_extparms.setdefault('ynormfunc', None)})
    task_params = dict([(k, getattr(opts, k)) if hasattr(opts, k) and getattr(opts, k) is not None else (k, v) for k, v in task_extparms.setdefault('mdlcfg', {}).items()])

    # Load model
    clf, prv_optimizer, resume, chckpnt = load_model(opts.resume)
    if opts.refresh:
        print('Refreshing and saving the model with newest code...')
        try:
            save_model(clf, prv_optimizer, '%s_%s.pth' % (opts.task, opts.model))
        except Exception as e:
            print(e)
    prv_task_params = copy.deepcopy(clf.task_params)
    # Update parameters
    clf.update_params(task_params=task_params)
    clf.mlt_trnsfmr = False
    if (use_gpu): clf = _handle_model(clf, dev_id=dev_id, distrb=opts.distrb)
    model = config.shell(clf, tokenizer=tokenizer, encode_func=encode_func, transforms=trsfms, transforms_kwargs=trsfms_kwargs, special_tknids_args=special_tknids_args, pad_val=task_extparms.setdefault('xpad_val', 0))

    # Prepare dev and test sets
    print('Dataset path: %s' % os.path.join(DATA_PATH, task_path))
    del ds_kwargs['ynormfunc']
    dev_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'dev.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else clf.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): dev_ds = MaskedLMIterDataset(dev_ds) if isinstance(dev_ds, BaseIterDataset) else MaskedLMDataset(dev_ds)
    dev_loader = DataLoader(dev_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)
    test_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'test.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else clf.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): test_ds = MaskedLMIterDataset(test_ds) if isinstance(test_ds, BaseIterDataset) else MaskedLMDataset(test_ds)
    test_loader = DataLoader(test_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)

    lbnotes_fname, lb_embd_fname, lb_embd_raw_fname = 'lbnotes.csv', 'onto_embd.pkl', 'onto_embd_raw.pkl'
    if os.path.exists(os.path.join(os.path.pardir, lb_embd_fname)):
        with open(os.path.join(os.path.pardir, lb_embd_fname), 'rb') as fd:
            lbembd = pickle.load(fd)
    else:
        lbnotes = pd.read_csv(os.path.join(os.path.pardir, lbnotes_fname), sep='\t', index_col='id')
        lb_embd_raw_fname = './onto_embd_raw.pkl'
        if os.path.exists(os.path.join(os.path.pardir, lb_embd_raw_fname)):
            with open(os.path.join(os.path.pardir, lb_embd_raw_fname), 'rb') as fd:
                lbembd_raw = pickle.load(fd)
        else:
            lb_corpus = lbnotes['text'].tolist()
            lbembd_raw = model(lb_corpus, embedding_mode=True, bsize=opts.bsize).numpy()
            if opts.do_tfidf:
                from sklearn.feature_extraction.text import TfidfVectorizer
                lb_vctrz = TfidfVectorizer()
                lb_X = lb_vctrz.fit_transform(lb_corpus)
                lbembd_raw = np.concatenate((lbembd_raw, lb_X), axis=1)
            if opts.do_bm25:
                from gensim.summarization.bm25 import get_bm25_weights
                lb_bm25_X = np.array(get_bm25_weights(lb_corpus, n_jobs=opts.np))
                lbembd_raw = np.concatenate((lbembd_raw, lb_bm25_X), axis=1)
            with open(lb_embd_raw_fname, 'wb') as fd:
                pickle.dump(lbembd_raw, fd)
        lbnotes['embedding'] = [x for x in lbembd_raw]
        lbembd = {}
        # One label may has multiple notes
        for gid, grp in lbnotes['embedding'].groupby('id'):
            lbembd[gid] = grp.mean(axis=0) # averaged embedding
        with open(lb_embd_fname, 'wb') as fd:
            pickle.dump(lbembd, fd)

    # Build label embedding index
    lb_embd_idx_fname = 'onto_embd_idx.pkl'
    if os.path.exists(os.path.join(os.path.pardir, lb_embd_idx_fname)):
        with open(os.path.join(os.path.pardir, lb_embd_idx_fname), 'rb') as fd:
            lbembd_idx, binlb, binlbr = pickle.load(fd)
    else:
        import faiss
        binlb = dict([(k, i) for i, k in enumerate(lbembd.keys())])
        binlbr = dict([(v, k) for k, v in binlb.items()])
        dimension = next(x for x in lbembd.values()).shape[0]
        lbembd_idx = faiss.IndexFlatL2(dimension)
        lbembd_idx.add(np.stack(lbembd.values()))
        # with open(lb_embd_idx_fname, 'wb') as fd:
        #     pickle.dump((lbembd_idx, binlb, binlbr), fd)
    lbembd_idx = faiss.index_cpu_to_all_gpus(lbembd_idx)

    import scipy.spatial.distance as spdist
    embd_clf = EmbeddingHead(clf)
    for prefix, df in zip(['dev', 'test'], [dev_ds.df, test_ds.df]):
        # Calculate the doc/sentence embeddings
        clf_h_cache_fname = '%s_clf_h.pkl' % prefix
        if os.path.exists(os.path.join(os.path.pardir, clf_h_cache_fname)):
            with open(os.path.join(os.path.pardir, clf_h_cache_fname), 'rb') as fd:
                clf_h = pickle.load(fd)
        else:
            txt_corpus = df['text'].tolist()
            clf_h = model(txt_corpus, embedding_mode=True, bsize=opts.bsize).numpy()
            if opts.do_tfidf:
                from sklearn.feature_extraction.text import TfidfVectorizer
                txt_vctrz = TfidfVectorizer()
                txt_X = txt_vctrz.fit_transform(txt_corpus)
                clf_h = np.concatenate((clf_h, txt_X), axis=1)
            if opts.do_bm25:
                from gensim.summarization.bm25 import get_bm25_weights
                txt_bm25_X = np.array(get_bm25_weights(txt_corpus, n_jobs=opts.np))
                clf_h = np.concatenate((clf_h, txt_bm25_X), axis=1)
            with open(clf_h_cache_fname, 'wb') as fd:
                pickle.dump(clf_h, fd)

        # Search the topk similar labels
        D, I = lbembd_idx.search(clf_h, opts.topk)
        cand_preds = [[binlbr[idx] for idx in indices] for indices in I]
        cand_lbs = list(set(itertools.chain.from_iterable(cand_preds)))
        cand_lbs_idx = dict([(lb, i) for i, lb in enumerate(cand_lbs)])
        cand_embds = np.stack([lbembd[lb] for lb in cand_lbs])
        pair_indices = [(i, cand_lbs_idx[lb]) for i, j in zip(range(len(clf_h)), range(len(cand_preds))) for lb in cand_preds[j]]
        ds = EmbeddingPairDataset(clf_h, cand_embds, pair_indices)
        ds_loader = DataLoader(ds, batch_size=opts.bsize, shuffle=False, sampler=None, num_workers=opts.np, drop_last=False)
        preds = []
        for step, batch in enumerate(tqdm(ds_loader, desc='[Totally %i pairs] Predicting pairs of embeddings' % len(ds))):
            embd_pairs = batch[0].to('cuda') if use_gpu else batch[0]
            embd_pairs = [embd_pairs[:,x,:] for x in [0,1]]
            logits = embd_clf(embd_pairs)
            prob = torch.sigmoid(logits).data.view(-1)
            pred = (prob > (embd_clf.thrshld if opts.do_thrshld else opts.pthrshld)).int()
            preds.extend(pred.view(-1).detach().cpu().tolist())
        orig_idx, pred_lb = zip(*[(pidx[0], cand_lbs[pidx[1]]) for pidx, pred_val in zip(pair_indices, preds) if pred_val > 0])
        pred_df = pd.DataFrame([(df.index[gid], ';'.join(grp['label'].tolist())) for gid, grp in pd.DataFrame({'index':orig_idx, 'label':pred_lb}).groupby('index')], columns=['id', 'preds']).set_index('id')
        filled_df = df.merge(pred_df, how='left', left_index=True, right_index=True)
        filled_df.to_csv('%s_preds.csv' % prefix, sep='\t')

        # Calculate the relations between the doc/sentence embedding and the true label embeddings
        angular_sim_list, correl_list, plot_data = [[] for x in range(3)]
        for i in range(df.shape[0]):
            angular_sims, correls = [], []
            lbs = df.iloc[i]['labels'].split(';')
            for lb in lbs:
                if lb in lbembd:
                    angular_sim, correlation = 1 - np.arccos(1 - spdist.cosine(clf_h[i], lbembd[lb])) / np.pi, 1 - spdist.correlation(clf_h[i], lbembd[lb])
                    _, _ = angular_sims.append('%.2f' % angular_sim), correls.append('%.2f' % correlation)
                    plot_data.append((1, angular_sim, correlation))
                else:
                    _, _ = angular_sims.append('N/A'), correls.append('N/A')
            _, _ = angular_sim_list.append(';'.join(angular_sims)), correl_list.append(';'.join(correls))
            neg_lbs = list(set(lbembd.keys()) - set(lbs))
            neg_idx = np.random.choice(range(len(neg_lbs)), len(lbs))
            for neg_lb in [neg_lbs[idx] for idx in neg_idx]:
                angular_sim, correlation = 1 - np.arccos(1 - spdist.cosine(clf_h[i], lbembd[neg_lb])) / np.pi, 1 - spdist.correlation(clf_h[i], lbembd[neg_lb])
                plot_data.append((0, angular_sim, correlation))
        df['angular_sim'], df['correlation'] = angular_sim_list, correl_list
        df.to_csv('%s_embd.csv' % prefix, sep='\t')
        with open('%s_plot_data.pkl' % prefix, 'wb') as fd:
            pickle.dump(plot_data, fd)


def simsearch(dev_id=None):
    print('### Similarity Search Mode ###')
    config_kwargs = dict([(k, v) for k, v in opts.__dict__.items() if not k.startswith('_') and k not in set(['dataset', 'model', 'template']) and v is not None and type(v) is not function])
    config = Configurable(opts.task, opts.model, wsdir=PAR_DIR, sc=SC, **config_kwargs)
    dict_fname, term_embd_fname = opts.corpus if opts.corpus else 'dict_df.csv', 'term_embd.pkl'
    if os.path.exists(os.path.join(os.path.pardir, term_embd_fname)):
        with open(os.path.join(os.path.pardir, term_embd_fname), 'rb') as fd:
            term_embd, term_labels, txt_vctrz, char_vctrz, ftmdl = pickle.load(fd)
    else:
        dict_df = pd.read_csv(os.path.join(os.path.pardir, dict_fname), sep='\t').set_index('id')
        term_labels, term_texts, term_types = zip(*[(idx, v, k) for idx, row in dict_df.iterrows() for k, v in row.items() if v is not np.nan])
        print('Corpus size from file %s: %i' % (dict_fname, dict_df.shape[0]))
        txt2vec_output = _txt2vec(term_texts, config, None, use_tfidf=opts.cfg.setdefault('use_tfidf', False), ftdecomp=None, saved_path=os.path.pardir, prefix='dict', **opts.cfg.setdefault('txt2vec_kwargs', {}))
        term_embd, txt_vctrz, char_vctrz, ftmdl = txt2vec_output[:4]
        feature_weights = txt2vec_output[4:]
        with open(term_embd_fname, 'wb') as fd:
            pickle.dump((term_embd, term_labels, txt_vctrz, char_vctrz, ftmdl), fd, protocol=4)

    # Build dictionary embedding index
    dict_embd_idx_fname = 'dict_embd_idx.pkl'
    if os.path.exists(os.path.join(os.path.pardir, dict_embd_idx_fname)):
        with open(os.path.join(os.path.pardir, dict_embd_idx_fname), 'rb') as fd:
            dict_embd_idx = pickle.load(fd)
    else:
        import faiss
        dimension = term_embd.shape[1]
        print('Building faiss index with dimension %i' % dimension)
        # dict_embd_idx = faiss.IndexFlatL2(dimension)
        from bionlp.util.math import VectorDB
        # dict_embd_idx = VectorDB(metric=lambda x, y: np.sqrt(np.sum(np.square(x*y-y)))/np.sqrt(np.sum(np.square(y))), feature_weights=feature_weights)
        dict_embd_idx = VectorDB(metric=lambda x, y: np.sum(np.abs(x*y-y))/np.sum(y), feature_weights=feature_weights)
        dict_embd_idx.add(term_embd.astype('float32'))
        # with open(dict_embd_idx_fname, 'wb') as fd:
        #     pickle.dump((dict_embd_idx), fd)
    # dict_embd_idx = faiss.index_cpu_to_all_gpus(dict_embd_idx)

    dev_df, test_df = pd.read_csv(os.path.join(DATA_PATH, opts.input, 'dev.%s' % opts.fmt), sep='\t', index_col='id'), pd.read_csv(os.path.join(DATA_PATH, opts.input, 'test.%s' % opts.fmt), sep='\t')
    w2v_cache = None
    for prefix, df in zip(['dev', 'test'], [dev_df, test_df]):
        txt_corpus = df['text'].tolist()
        enriched_texts, w2v_cache = enrich_txt_by_w2v(txt_corpus, w2v_model=opts.cfg.setdefault('w2v_model', None), w2v_cache=w2v_cache, topk=opts.cfg.setdefault('txt2vec_kwargs', {}).setdefault('w2v_topk', 10))
        clf_h, _, _, _ = _txt2vec(enriched_texts, config, None, txt_vctrz=txt_vctrz, char_vctrz=char_vctrz, use_tfidf=opts.cfg.setdefault('use_tfidf', False), ftdecomp=None, saved_path=os.path.pardir, prefix=prefix, **opts.cfg.setdefault('txt2vec_kwargs', {}))

        # Search the topk similar labels
        print('Searching dataset %s with size: %s...' % (prefix, str(clf_h.shape)))
        clf_h = clf_h.astype('float32')
        D, I = dict_embd_idx.search(clf_h[:,:dimension] if clf_h.shape[1] >= dimension else np.hstack((clf_h, np.zeros((clf_h.shape[0], dimension-clf_h.shape[1]), dtype=clf_h.dtype))), opts.topk, n_jobs=opts.np)
        cand_preds = [[term_labels[idx] for idx in idxs] for idxs in I]
        cand_lbs = [sorted(set(lbs)) for lbs in cand_preds]
        df['preds'] = [';'.join(lbs) for lbs in cand_lbs]
        df.to_csv('%s_preds.csv' % prefix, sep='\t')

		# Evaluation
        from sklearn.preprocessing import MultiLabelBinarizer
        from bionlp.util import math as imath
        true_lbs = [lbs_str.split(';') if type(lbs_str) is str and not lbs_str.isspace() else [] for lbs_str in df['labels']]
        mlb = MultiLabelBinarizer()
        mlb = mlb.fit(true_lbs + cand_lbs)
        lbs = mlb.transform(true_lbs)
        pred_lbs = mlb.transform(cand_lbs)
        print('exmp-precision: %.3f' % imath.exmp_precision(lbs, pred_lbs) + '\texmp-recall: %.3f' % imath.exmp_recall(lbs, pred_lbs) + '\texmp-f1-score: %.3f' % imath.exmp_fscore(lbs, pred_lbs) + '\n')


def simsearch_sentembd(dev_id=None):
    print('### Similarity Search Mode using sentence embedding ###')
    orig_task = opts.task
    opts.task = opts.task + '_simsearch'
    # Prepare model related meta data
    mdl_name = opts.model.split('_')[0].lower().replace(' ', '_')
    common_cfg = cfgr('validate', 'common')
    pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
    config_kwargs = dict([(k, v) for k, v in opts.__dict__.items() if not k.startswith('_') and k not in set(['dataset', 'model', 'template']) and v is not None and type(v) is not function])
    config = Configurable(opts.task, mdl_name, common_cfg=common_cfg, wsdir=PAR_DIR, sc=SC, **config_kwargs)
    params = pr('LM', config.lm_params) if mdl_name != 'none' else {}
    use_gpu = dev_id is not None
    encode_func = config.encode_func
    tokenizer = config.tknzr.from_pretrained(params['pretrained_vocab_path'] if 'pretrained_vocab_path' in params else config.lm_mdl_name) if config.tknzr else None
    task_type = 'sentsim' #TASK_TYPE_MAP[opts.task]
    spcl_tkns = config.lm_tknz_extra_char if config.lm_tknz_extra_char else ['_@_', ' _$_', ' _#_']
    special_tkns = (['start_tknids', 'clf_tknids', 'delim_tknids'], spcl_tkns[:3]) if task_type in ['entlmnt', 'sentsim'] else (['start_tknids', 'clf_tknids'], spcl_tkns[:2])
    special_tknids = _adjust_encoder(mdl_name, tokenizer, config, special_tkns[1], ret_list=True)
    special_tknids_args = dict(zip(special_tkns[0], special_tknids))
    task_trsfm_kwargs = dict(list(zip(special_tkns[0], special_tknids))+[('model',opts.model), ('sentsim_func', opts.sentsim_func), ('seqlen',opts.maxlen)])
    # Prepare task related meta data
    task_path, task_dstype, task_cols, task_trsfm, task_extrsfm, task_extparms = opts.input if opts.input and os.path.isdir(os.path.join(DATA_PATH, opts.input)) else config.task_path, config.task_ds, config.task_col, config.task_trsfm, config.task_ext_trsfm, config.task_ext_params
    trsfms = ([] if hasattr(config, 'embed_type') and config.embed_type else task_extrsfm[0]) + (task_trsfm[0] if len(task_trsfm) > 0 else [])
    trsfms_kwargs = ([] if hasattr(config, 'embed_type') and config.embed_type else ([{'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}] if config.task_type=='nmt' else [{'seqlen':opts.maxlen, 'trimlbs':task_extparms.setdefault('trimlbs', False), 'required_special_tkns':['start_tknids', 'clf_tknids', 'delim_tknids'] if task_type in ['entlmnt', 'sentsim'] and (task_extparms.setdefault('sentsim_func', None) is None or not mdl_name.startswith('bert')) else ['start_tknids', 'clf_tknids'], 'special_tkns':special_tknids_args}, task_trsfm_kwargs, {'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}])) + (task_trsfm[1] if len(task_trsfm) >= 2 else [{}] * len(task_trsfm[0]))
    ds_kwargs = {'sampw':opts.sample_weights, 'sampfrac':opts.sampfrac}
    ds_kwargs.update({'ynormfunc':task_extparms.setdefault('ynormfunc', None)})
    task_params = dict([(k, getattr(opts, k)) if hasattr(opts, k) and getattr(opts, k) is not None else (k, v) for k, v in task_extparms.setdefault('mdlcfg', {}).items()])

    # Load model
    clf, prv_optimizer, resume, chckpnt = load_model(opts.resume)
    if opts.refresh:
        print('Refreshing and saving the model with newest code...')
        try:
            save_model(clf, prv_optimizer, '%s_%s.pth' % (opts.task, opts.model))
        except Exception as e:
            print(e)
    prv_task_params = copy.deepcopy(clf.task_params)
    # Update parameters
    clf.update_params(task_params=task_params, sample_weights=False)
    clf.mlt_trnsfmr = False
    if (use_gpu): clf = _handle_model(clf, dev_id=dev_id, distrb=opts.distrb)
    model = config.shell(clf, tokenizer=tokenizer, encode_func=encode_func, transforms=trsfms, transforms_kwargs=trsfms_kwargs, special_tknids_args=special_tknids_args, pad_val=task_extparms.setdefault('xpad_val', 0))

    # Prepare dev and test sets
    print('Dataset path: %s' % os.path.join(DATA_PATH, task_path))
    del ds_kwargs['ynormfunc']
    dev_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'dev.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else clf.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): dev_ds = MaskedLMIterDataset(dev_ds) if isinstance(dev_ds, BaseIterDataset) else MaskedLMDataset(dev_ds)
    dev_loader = DataLoader(dev_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)
    test_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'test.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else clf.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): test_ds = MaskedLMIterDataset(test_ds) if isinstance(test_ds, BaseIterDataset) else MaskedLMDataset(test_ds)
    test_loader = DataLoader(test_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)

    corpus_fnames, corpus_embd_fname, corpus_embd_raw_fname = opts.corpus.split(SC) if opts.corpus else ['corpus_df.csv'], 'corpus_embd.pkl', 'corpus_embd_raw.pkl'
    if os.path.exists(os.path.join(os.path.pardir, corpus_embd_fname)):
        with open(os.path.join(os.path.pardir, corpus_embd_fname), 'rb') as fd:
            corpus_embd, labels, txt_vctrz, char_vctrz, ftmdl = pickle.load(fd)
    else:
        corpus_df = pd.concat([pd.read_csv(os.path.join(os.path.pardir, fname), sep='\t') for fname in corpus_fnames], axis=0, ignore_index=True).set_index('id')
        labels = corpus_df['labels']
        corpus_texts = corpus_df['text'].tolist()
        corpus_embd_raw_fname = './corpus_embd_raw.pkl'
        print('Corpus size from file(s) %s: %i' % (corpus_fnames, corpus_df.shape[0]))
        if os.path.exists(os.path.join(os.path.pardir, corpus_embd_raw_fname)):
            with open(os.path.join(os.path.pardir, corpus_embd_raw_fname), 'rb') as fd:
                corpus_embd_raw = pickle.load(fd)
        else:
            corpus_embd_raw = model(corpus_texts, embedding_mode=True, bsize=opts.bsize).numpy()
            with open(corpus_embd_raw_fname, 'wb') as fd:
                pickle.dump(corpus_embd_raw, fd)
        corpus_embd_raw, txt_vctrz, char_vctrz, ftmdl = _txt2vec(corpus_texts, config, corpus_embd_raw if opts.cfg.setdefault('clf_h', True) else None, use_tfidf=opts.cfg.setdefault('use_tfidf', True), ftdecomp=opts.cfg.setdefault('ftdecomp', 'pca'), n_components=opts.cfg.setdefault('n_components', 768), saved_path=os.path.pardir, prefix='corpus', **opts.cfg.setdefault('txt2vec_kwargs', {}))
        corpus_df['embedding'] = [np.array(x).reshape((-1,)) for x in corpus_embd_raw]
        corpus_embd = {}
        # One label may has multiple notes
        for gid, grp in corpus_df['embedding'].groupby('id'):
            corpus_embd[gid] = grp.mean(axis=0) # averaged embedding
        with open(corpus_embd_fname, 'wb') as fd:
            pickle.dump((corpus_embd, labels, txt_vctrz, char_vctrz, ftmdl), fd)

    # Build corpus embedding index
    corpus_embd_idx_fname = 'corpus_embd_idx.pkl'
    if os.path.exists(os.path.join(os.path.pardir, corpus_embd_idx_fname)):
        with open(os.path.join(os.path.pardir, corpus_embd_idx_fname), 'rb') as fd:
            corpus_embd_idx, indices, rindices = pickle.load(fd)
    else:
        import faiss
        indices = dict([(k, i) for i, k in enumerate(corpus_embd.keys())])
        rindices = dict([(v, k) for k, v in indices.items()])
        dimension = next(x for x in corpus_embd.values()).shape[0]
        print('Building faiss index with dimension %i' % dimension)
        corpus_embd_idx = faiss.IndexFlatL2(dimension)
        corpus_embd_idx.add(np.stack(corpus_embd.values()).astype('float32'))
        # with open(corpus_embd_idx_fname, 'wb') as fd:
        #     pickle.dump((corpus_embd_idx, indices, rindices), fd)
    # corpus_embd_idx = faiss.index_cpu_to_all_gpus(corpus_embd_idx)

    embd_clf = EmbeddingHead(clf)
    for prefix, df in zip(['dev', 'test'], [dev_ds.df, test_ds.df]):
        txt_corpus = df['text'].tolist()
        # Calculate the doc/sentence embeddings
        clf_h_cache_fname = '%s_clf_h.pkl' % prefix
        if os.path.exists(os.path.join(os.path.pardir, clf_h_cache_fname)):
            with open(os.path.join(os.path.pardir, clf_h_cache_fname), 'rb') as fd:
                clf_h = pickle.load(fd)
        else:
            clf_h = model(txt_corpus, embedding_mode=True, bsize=opts.bsize).numpy()
            with open(clf_h_cache_fname, 'wb') as fd:
                pickle.dump(clf_h, fd)
        clf_h, _, _, _ = _txt2vec(txt_corpus, config, clf_h if opts.cfg.setdefault('clf_h', True) else None, txt_vctrz=txt_vctrz, char_vctrz=char_vctrz, use_tfidf=opts.cfg.setdefault('use_tfidf', True), ftdecomp=opts.cfg.setdefault('ftdecomp', 'pca'), ftmdl=ftmdl, n_components=opts.cfg.setdefault('n_components', 768), saved_path=os.path.pardir, prefix=prefix, **opts.cfg.setdefault('txt2vec_kwargs', {}))

        # Search the topk similar labels
        print('Searching dataset %s with size: %s...' % (prefix, str(clf_h.shape)))
        clf_h = clf_h.astype('float32')
        D, I = corpus_embd_idx.search(clf_h[:,:dimension] if clf_h.shape[1] >= dimension else np.hstack((clf_h, np.zeros((clf_h.shape[0], dimension-clf_h.shape[1]), dtype=clf_h.dtype))), opts.topk)
        cand_preds = [[labels.iloc[idx].split(';') for idx in idxs] for idxs in I]
        cand_lbs = [sorted(set(itertools.chain.from_iterable(lbs))) for lbs in cand_preds]
        df['preds'] = [';'.join(lbs) for lbs in cand_lbs]
        df.to_csv('%s_preds.csv' % prefix, sep='\t')

		# Evaluation
        from sklearn.preprocessing import MultiLabelBinarizer
        from bionlp.util import math as imath
        true_lbs = [lbs_str.split(';') if type(lbs_str) is str and not lbs_str.isspace() else [] for lbs_str in df['labels']]
        mlb = MultiLabelBinarizer()
        mlb = mlb.fit(true_lbs + cand_lbs)
        lbs = mlb.transform(true_lbs)
        pred_lbs = mlb.transform(cand_lbs)
        print('exmp-precision: %.3f' % imath.exmp_precision(lbs, pred_lbs) + '\texmp-recall: %.3f' % imath.exmp_recall(lbs, pred_lbs) + '\texmp-f1-score: %.3f' % imath.exmp_fscore(lbs, pred_lbs) + '\n')


def rerank(dev_id=None):
    print('### Re-rank Mode ###')
    orig_task = opts.task
    opts.task = opts.task + '_rerank'
    # Prepare model related meta data
    mdl_name = opts.model.split('_')[0].lower().replace(' ', '_')
    common_cfg = cfgr('validate', 'common')
    pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
    config_kwargs = dict([(k, v) for k, v in opts.__dict__.items() if not k.startswith('_') and k not in set(['dataset', 'model', 'template']) and v is not None and type(v) is not function])
    config = Configurable(opts.task, mdl_name, common_cfg=common_cfg, wsdir=PAR_DIR, sc=SC, **config_kwargs)
    params = pr('LM', config.lm_params) if mdl_name != 'none' else {}
    use_gpu = dev_id is not None
    encode_func = config.encode_func
    tokenizer = config.tknzr.from_pretrained(params['pretrained_vocab_path'] if 'pretrained_vocab_path' in params else config.lm_mdl_name) if config.tknzr else None
    task_type = 'entlmnt' #TASK_TYPE_MAP[opts.task]
    spcl_tkns = config.lm_tknz_extra_char if config.lm_tknz_extra_char else ['_@_', ' _$_', ' _#_']
    special_tkns = (['start_tknids', 'clf_tknids', 'delim_tknids'], spcl_tkns[:3]) if task_type in ['entlmnt', 'sentsim'] else (['start_tknids', 'clf_tknids'], spcl_tkns[:2])
    special_tknids = _adjust_encoder(mdl_name, tokenizer, config, special_tkns[1], ret_list=True)
    special_tknids_args = dict(zip(special_tkns[0], special_tknids))
    task_trsfm_kwargs = dict(list(zip(special_tkns[0], special_tknids))+[('model',opts.model), ('sentsim_func', opts.sentsim_func), ('seqlen',opts.maxlen)])
    # Prepare task related meta data
    task_path, task_dstype, task_cols, task_trsfm, task_extrsfm, task_extparms = opts.input if opts.input and os.path.isdir(os.path.join(DATA_PATH, opts.input)) else config.task_path, config.task_ds, config.task_col, config.task_trsfm, config.task_ext_trsfm, config.task_ext_params
    trsfms = ([] if hasattr(config, 'embed_type') and config.embed_type else task_extrsfm[0]) + (task_trsfm[0] if len(task_trsfm) > 0 else [])
    trsfms_kwargs = ([] if hasattr(config, 'embed_type') and config.embed_type else ([{'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}] if config.task_type=='nmt' else [{'seqlen':opts.maxlen, 'trimlbs':task_extparms.setdefault('trimlbs', False), 'required_special_tkns':['start_tknids', 'clf_tknids', 'delim_tknids'] if task_type in ['entlmnt', 'sentsim'] and (task_extparms.setdefault('sentsim_func', None) is None or not mdl_name.startswith('bert')) else ['start_tknids', 'clf_tknids'], 'special_tkns':special_tknids_args}, task_trsfm_kwargs, {'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}])) + (task_trsfm[1] if len(task_trsfm) >= 2 else [{}] * len(task_trsfm[0]))
    ds_kwargs = {'ynormfunc':task_extparms.setdefault('ynormfunc', None)}
    ds_kwargs.update(dict((k, task_extparms[k]) for k in ['origlb', 'locallb', 'lbtxt', 'neglbs', 'reflb', 'sent_mode'] if k in task_extparms))
    if task_dstype in [OntoDataset, OntoIterDataset]:
        ds_kwargs['onto_fpath'] = opts.onto if opts.onto else task_extparms.setdefault('onto_fpath', 'onto.csv')
        ds_kwargs['onto_col'] = task_cols['ontoid']
    task_params = dict([(k, getattr(opts, k)) if hasattr(opts, k) and getattr(opts, k) is not None else (k, v) for k, v in task_extparms.setdefault('mdlcfg', {}).items()])

    # Load model
    clf, prv_optimizer, resume, chckpnt = load_model(opts.resume)
    if opts.refresh:
        print('Refreshing and saving the model with newest code...')
        try:
            save_model(clf, prv_optimizer, '%s_%s.pth' % (opts.task, opts.model))
        except Exception as e:
            print(e)
    prv_task_params = copy.deepcopy(clf.task_params)
    # Update parameters
    clf.update_params(task_params=task_params, sample_weights=False)
    if (use_gpu): clf = _handle_model(clf, dev_id=dev_id, distrb=opts.distrb)

    # Prepare dev and test sets
    print('Dataset path: %s' % os.path.join(DATA_PATH, task_path))
    del ds_kwargs['ynormfunc']
    dev_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'dev.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms else clf.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, **ds_kwargs)
    if mdl_name.startswith('bert'): dev_ds = MaskedLMIterDataset(dev_ds) if isinstance(dev_ds, BaseIterDataset) else MaskedLMDataset(dev_ds)
    dev_loader = DataLoader(dev_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)
    test_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'test.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer=tokenizer, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms else clf.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, **ds_kwargs)
    if mdl_name.startswith('bert'): test_ds = MaskedLMIterDataset(test_ds) if isinstance(test_ds, BaseIterDataset) else MaskedLMDataset(test_ds)
    test_loader = DataLoader(test_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)

    # Evaluation
    eval(clf, dev_loader, dev_ds.binlbr, special_tknids_args, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='dev', mdl_name=opts.model, use_gpu=use_gpu, devq=dev_id, distrb=opts.distrb, ignored_label=task_extparms.setdefault('ignored_label', None))
    eval(clf, test_loader, test_ds.binlbr, special_tknids_args, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='test', mdl_name=opts.model, use_gpu=use_gpu, devq=dev_id, distrb=opts.distrb, ignored_label=task_extparms.setdefault('ignored_label', None))
