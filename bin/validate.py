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

import os, sys, ast, random, logging
from optparse import OptionParser

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from transformers import get_linear_schedule_with_warmup

from bionlp.util import io

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))

sys.path.insert(0, PAR_DIR)

from modules.varclf import MultiClfTransformer

from util.config import *
from util.dataset import MaskedLMDataset, OntoDataset, BaseIterDataset, OntoIterDataset, MaskedLMIterDataset
from util.processor import _adjust_encoder
from util.trainer import train, eval
from util.func import _update_cfgs, gen_mdl, gen_clf, save_model, load_model

CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
SC=';;'
NUM_TRIM = 0

opts, args = {}, []
cfgr = None


def classify(dev_id=None):
    # Prepare model related meta data
    mdl_name = opts.model.lower().replace(' ', '_')
    common_cfg = cfgr('validate', 'common')
    pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
    config_kwargs = dict([(k, v) for k, v in opts.__dict__.items() if not k.startswith('_') and k not in set(['dataset', 'model', 'template']) and v is not None and not callable(v)])
    config = Configurable(opts.task, mdl_name, common_cfg=common_cfg, wsdir=PAR_DIR, sc=SC, **config_kwargs)
    params = pr('LM', config.lm_params) if mdl_name != 'none' else {}
    use_gpu = dev_id is not None
    encode_func = config.encode_func
    tokenizer = config.tknzr.from_pretrained(params['pretrained_vocab_path'] if 'pretrained_vocab_path' in params else config.lm_mdl_name) if config.tknzr else None
    task_type = config.task_type
    # spcl_tkns = LM_TKNZ_EXTRA_CHAR.setdefault(mdl_name, ['_@_', ' _$_', ' _#_'])
    spcl_tkns = config.lm_tknz_extra_char if config.lm_tknz_extra_char else ['_@_', ' _$_', ' _#_']
    special_tkns = (['start_tknids', 'clf_tknids', 'delim_tknids'], spcl_tkns[:3]) if task_type in ['entlmnt', 'sentsim'] else (['start_tknids', 'clf_tknids'], spcl_tkns[:2])
    special_tknids = _adjust_encoder(mdl_name, tokenizer, config, special_tkns[1], ret_list=True)
    special_tknids_args = dict(zip(special_tkns[0], special_tknids))
    task_trsfm_kwargs = dict(list(zip(special_tkns[0], special_tknids))+[('model',opts.model), ('sentsim_func', opts.sentsim_func), ('seqlen',opts.maxlen)])
    # Prepare task related meta data.
    task_path, task_dstype, task_cols, task_trsfm, task_extrsfm, task_extparms = opts.input if opts.input and os.path.isdir(os.path.join(DATA_PATH, opts.input)) else config.task_path, config.task_ds, config.task_col, config.task_trsfm, config.task_ext_trsfm, config.task_ext_params
    # trsfms = ([] if opts.model in LM_EMBED_MDL_MAP else task_extrsfm[0]) + (task_trsfm[0] if len(task_trsfm) > 0 else [])
    trsfms = ([] if hasattr(config, 'embed_type') and config.embed_type else task_extrsfm[0]) + (task_trsfm[0] if len(task_trsfm) > 0 else [])
    # trsfms_kwargs = ([] if opts.model in LM_EMBED_MDL_MAP else ([{'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}] if TASK_TYPE_MAP[opts.task]=='nmt' else [{'seqlen':opts.maxlen, 'trimlbs':task_extparms.setdefault('trimlbs', False), 'required_special_tkns':['start_tknids', 'clf_tknids', 'delim_tknids'] if task_type in ['entlmnt', 'sentsim'] and (task_extparms.setdefault('sentsim_func', None) is None or not mdl_name.startswith('bert')) else ['start_tknids', 'clf_tknids'], 'special_tkns':special_tknids_args}, task_trsfm_kwargs, {'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}])) + (task_trsfm[1] if len(task_trsfm) >= 2 else [{}] * len(task_trsfm[0]))
    trsfms_kwargs = ([] if hasattr(config, 'embed_type') and config.embed_type else ([{'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}] if config.task_type=='nmt' else [{'seqlen':opts.maxlen, 'trimlbs':task_extparms.setdefault('trimlbs', False), 'required_special_tkns':['start_tknids', 'clf_tknids', 'delim_tknids'] if task_type in ['entlmnt', 'sentsim'] and (task_extparms.setdefault('sentsim_func', None) is None or not mdl_name.startswith('bert')) else ['start_tknids', 'clf_tknids'], 'special_tkns':special_tknids_args}, task_trsfm_kwargs, {'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}])) + (task_trsfm[1] if len(task_trsfm) >= 2 else [{}] * len(task_trsfm[0]))
    ds_kwargs = {'sampw':opts.sample_weights, 'sampfrac':opts.sampfrac}
    if task_type == 'nmt':
        ds_kwargs.update({'lb_coding':task_extparms.setdefault('lb_coding', 'IOB')})
    elif task_type == 'entlmnt':
        ds_kwargs.update(dict((k, task_extparms[k]) for k in ['origlb', 'lbtxt', 'neglbs', 'reflb'] if k in task_extparms))
    elif task_type == 'sentsim':
        ds_kwargs.update({'ynormfunc':task_extparms.setdefault('ynormfunc', None)})
    if task_dstype in [OntoDataset, OntoIterDataset]:
        ds_kwargs['onto_fpath'] = opts.onto if opts.onto and os.path.exists(opts.onto) else task_extparms.setdefault('onto_fpath', 'onto.csv')
        ds_kwargs['onto_col'] = task_cols['ontoid']

    # Prepare data
    if (not opts.distrb or opts.distrb and hvd.rank() == 0): print('Dataset path: %s' % os.path.join(DATA_PATH, task_path))
    train_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'train.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer, config, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms else None, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): train_ds = MaskedLMIterDataset(train_ds) if isinstance(train_ds, BaseIterDataset) else MaskedLMDataset(train_ds)
    lb_trsfm = [x['get_lb'] for x in task_trsfm[1] if 'get_lb' in x]
    if hasattr(train_ds, 'df'):
        train_ds.df[task_cols['y']] = ['false' if lb is None or lb is np.nan or (type(lb) is str and lb.isspace()) else lb for lb in train_ds.df[task_cols['y']]]
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
        if not opts.distrb and type(dev_id) is list: class_weights = class_weights.repeat(len(dev_id))

    if opts.distrb:
        # Partition dataset among workers using DistributedSampler
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = DataLoader(train_ds, batch_size=opts.bsize, shuffle=sampler is None and opts.droplast, sampler=sampler, num_workers=opts.np, drop_last=opts.droplast)

    ext_params = dict([(k, getattr(opts, k)) if hasattr(opts, k) else (k, v) for k, v in config.clf_ext_params.items()])
    if hasattr(config, 'embed_type'): ext_params['embed_type'] = config.embed_type
    task_params = dict([(k, getattr(opts, k)) if hasattr(opts, k) and getattr(opts, k) is not None else (k, v) for k, v in task_extparms.setdefault('mdlcfg', {}).items()])
    if (not opts.distrb or opts.distrb and hvd.rank() == 0):
        print('Classifier hyper-parameters: %s' % ext_params)
        print('Classifier task-related parameters: %s' % task_params)
    if (opts.resume):
        # Load model
        clf, prv_optimizer, resume, chckpnt = load_model(opts.resume)
        if opts.refresh:
            print('Refreshing and saving the model with newest code...')
            try:
                if (not distrb or distrb and hvd.rank() == 0):
                    save_model(clf, prv_optimizer, '%s_%s.pth' % (opts.task, opts.model))
            except Exception as e:
                print(e)
        # Update parameters
        clf.update_params(task_params=task_params, **ext_params)
        if (use_gpu): clf = _handle_model(clf, dev_id=dev_id, distrb=opts.distrb)
        # optmzr_cls = OPTMZR_MAP.setdefault(opts.model.split('_')[0], (torch.optim.Adam, {}, None))
        optmzr_cls = config.optmzr if config.optmzr else (torch.optim.Adam, {}, None)
        optimizer = optmzr_cls[0](clf.parameters(), lr=opts.lr, weight_decay=opts.wdecay, **optmzr_cls[1]) if opts.optim == 'adam' else torch.optim.SGD(clf.parameters(), lr=opts.lr, momentum=0.9)
        if prv_optimizer: optimizer.load_state_dict(prv_optimizer.state_dict())
        training_steps = int(len(train_ds) / opts.bsize) if hasattr(train_ds, '__len__') else opts.trainsteps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(opts.wrmprop*training_steps), num_training_steps=training_steps) if not opts.noschdlr and len(optmzr_cls) > 2 and optmzr_cls[2] and optmzr_cls[2] == 'linwarm' else None
        if (not opts.distrb or opts.distrb and hvd.rank() == 0): print((optimizer, scheduler))
    else:
        # Build model
        lm_model = gen_mdl(mdl_name, config, pretrained=True if type(opts.pretrained) is str and opts.pretrained.lower() == 'true' else opts.pretrained, use_gpu=use_gpu, distrb=opts.distrb, dev_id=dev_id) if mdl_name != 'none' else None
        clf = gen_clf(opts.model, config, opts.encoder, lm_model=lm_model, constraints=opts.cnstrnts.split(',') if opts.cnstrnts else [], task_type=task_type, num_lbs=len(train_ds.binlb) if train_ds.binlb else 1, binlb=train_ds.binlb, mlt_trnsfmr=True if task_type in ['entlmnt', 'sentsim'] and task_params.setdefault('sentsim_func', None) is not None else False, task_params=task_params, use_gpu=use_gpu, distrb=opts.distrb, dev_id=dev_id, **ext_params)
        # optmzr_cls = OPTMZR_MAP.setdefault(opts.model.split('_')[0], (torch.optim.Adam, {}, None))
        optmzr_cls = config.optmzr if config.optmzr else (torch.optim.Adam, {}, None)
        optimizer = optmzr_cls[0](clf.parameters(), lr=opts.lr, weight_decay=opts.wdecay, **optmzr_cls[1]) if opts.optim == 'adam' else torch.optim.SGD(clf.parameters(), lr=opts.lr, momentum=0.9)
        training_steps = int(len(train_ds) / opts.bsize) if hasattr(train_ds, '__len__') else opts.trainsteps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opts.wrmprop, num_training_steps=training_steps) if not opts.noschdlr and len(optmzr_cls) > 2 and optmzr_cls[2] and optmzr_cls[2] == 'linwarm' else None
        if (not opts.distrb or opts.distrb and hvd.rank() == 0): print((optimizer, scheduler))

    if opts.distrb:
        # Add Horovod Distributed Optimizer
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=clf.named_parameters())
        # Broadcast parameters from rank 0 to all other processes.
        hvd.broadcast_parameters(clf.state_dict(), root_rank=0)

    # Training
    train(clf, optimizer, train_loader, config, special_tknids_args, scheduler, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), weights=class_weights, lmcoef=opts.lmcoef, clipmaxn=opts.clipmaxn, epochs=opts.epochs, earlystop=opts.earlystop, earlystop_delta=opts.es_delta, earlystop_patience=opts.es_patience, task_type=task_type, task_name=opts.task, mdl_name=opts.model, use_gpu=use_gpu, devq=dev_id, distrb=opts.distrb, resume=resume if opts.resume else {})

    if opts.distrb:
        if hvd.rank() == 0:
            clf = _handle_model(clf, dev_id=dev_id, distrb=False)
        else:
            return

    if opts.noeval: return
    dev_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'dev.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer, config, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else train_ds.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): dev_ds = MaskedLMIterDataset(train_ds) if isinstance(train_ds, BaseIterDataset) else MaskedLMDataset(dev_ds)
    dev_loader = DataLoader(dev_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)
    test_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'test.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer, config, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else train_ds.binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): test_ds = MaskedLMIterDataset(train_ds) if isinstance(train_ds, BaseIterDataset) else MaskedLMDataset(test_ds)
    test_loader = DataLoader(test_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)
    # print((train_ds.binlb, dev_ds.binlb, test_ds.binlb))

    # Evaluation
    eval(clf, dev_loader, config, dev_ds.binlbr, special_tknids_args, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='dev', mdl_name=opts.model, use_gpu=use_gpu, devq=dev_id, distrb=opts.distrb, ignored_label=task_extparms.setdefault('ignored_label', None))
    if opts.traindev: train(clf, optimizer, dev_loader, config, special_tknids_args, scheduler=scheduler, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), weights=class_weights, lmcoef=opts.lmcoef, clipmaxn=opts.clipmaxn, epochs=opts.epochs, earlystop=opts.earlystop, earlystop_delta=opts.es_delta, earlystop_patience=opts.es_patience, task_type=task_type, task_name=opts.task, mdl_name=opts.model, use_gpu=use_gpu, devq=dev_id, distrb=opts.distrb)
    eval(clf, test_loader, config, test_ds.binlbr, special_tknids_args, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='test', mdl_name=opts.model, use_gpu=use_gpu, devq=dev_id, distrb=opts.distrb, ignored_label=task_extparms.setdefault('ignored_label', None))


def mltphz_clf(dev_id=None):
    cfg_kwargs = opts.cfg
    common_cfg_kwargs = dict([(k, v) for k, v in cfg_kwargs.items() if k != 'phases'])
    _update_cfgs(common_cfg_kwargs)
    for cfgs in cfg_kwargs['phases']:
        _update_cfgs(cfgs)
        classify(dev_id=dev_id)
        setattr(opts, 'resume', '%s_%s.pth' % (opts.task, opts.model))


def multi_clf(dev_id=None):
    '''Train multiple classifiers and use them to predict multiple set of labels'''
    import inflect
    from bionlp.util import fs
    iflteng = inflect.engine()

    print('### Multi Classifier Head Mode ###')
    # Prepare model related meta data
    mdl_name = opts.model.lower().replace(' ', '_')
    common_cfg = cfgr('validate', 'common')
    pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
    config_kwargs = dict([(k, v) for k, v in opts.__dict__.items() if not k.startswith('_') and k not in set(['dataset', 'model', 'template']) and v is not None and type(v) is not function])
    config = Configurable(opts.task, mdl_name, common_cfg=common_cfg, wsdir=PAR_DIR, sc=SC, **config_kwargs)
    params = pr('LM', config.lm_params) if mdl_name != 'none' else {}
    use_gpu = dev_id is not None
    encode_func = config.encode_func
    tokenizer = config.tknzr.from_pretrained(params['pretrained_vocab_path'] if 'pretrained_vocab_path' in params else config.lm_mdl_name) if config.tknzr else None
    task_type = config.task_type
	# spcl_tkns = LM_TKNZ_EXTRA_CHAR.setdefault(mdl_name, ['_@_', ' _$_', ' _#_'])
    spcl_tkns = config.lm_tknz_extra_char if config.lm_tknz_extra_char else ['_@_', ' _$_', ' _#_']
    special_tkns = (['start_tknids', 'clf_tknids', 'delim_tknids'], spcl_tkns[:3]) if task_type in ['entlmnt', 'sentsim'] else (['start_tknids', 'clf_tknids'], spcl_tkns[:2])
    special_tknids = _adjust_encoder(mdl_name, tokenizer, config, special_tkns[1], ret_list=True)
    special_tknids_args = dict(zip(special_tkns[0], special_tknids))
    task_trsfm_kwargs = dict(list(zip(special_tkns[0], special_tknids))+[('model',opts.model), ('sentsim_func', opts.sentsim_func), ('seqlen',opts.maxlen)])
    # Prepare task related meta data.
    task_path, task_dstype, task_cols, task_trsfm, task_extrsfm, task_extparms = opts.input if opts.input and os.path.isdir(os.path.join(DATA_PATH, opts.input)) else config.task_path, config.task_ds, config.task_col, config.task_trsfm, config.task_ext_trsfm, config.task_ext_params
    # trsfms = ([] if opts.model in LM_EMBED_MDL_MAP else task_extrsfm[0]) + (task_trsfm[0] if len(task_trsfm) > 0 else [])
    trsfms = ([] if hasattr(config, 'embed_type') and config.embed_type else task_extrsfm[0]) + (task_trsfm[0] if len(task_trsfm) > 0 else [])
    # trsfms_kwargs = ([] if opts.model in LM_EMBED_MDL_MAP else ([{'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}] if TASK_TYPE_MAP[opts.task]=='nmt' else [{'seqlen':opts.maxlen, 'trimlbs':task_extparms.setdefault('trimlbs', False), 'special_tkns':special_tknids_args}, task_trsfm_kwargs, {'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}])) + (task_trsfm[1] if len(task_trsfm) >= 2 else [{}] * len(task_trsfm[0]))
    trsfms_kwargs = ([] if hasattr(config, 'embed_type') and config.embed_type else ([{'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}] if config.task_type=='nmt' else [{'seqlen':opts.maxlen, 'trimlbs':task_extparms.setdefault('trimlbs', False), 'required_special_tkns':['start_tknids', 'clf_tknids', 'delim_tknids'] if task_type in ['entlmnt', 'sentsim'] and (task_extparms.setdefault('sentsim_func', None) is None or not mdl_name.startswith('bert')) else ['start_tknids', 'clf_tknids'], 'special_tkns':special_tknids_args}, task_trsfm_kwargs, {'seqlen':opts.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}])) + (task_trsfm[1] if len(task_trsfm) >= 2 else [{}] * len(task_trsfm[0]))
    ds_kwargs = {'sampw':opts.sample_weights, 'sampfrac':opts.sampfrac}
    if task_type == 'nmt':
        ds_kwargs.update({'lb_coding':task_extparms.setdefault('lb_coding', 'IOB')})
    elif task_type == 'entlmnt':
        ds_kwargs.update(dict((k, task_extparms[k]) for k in ['origlb', 'lbtxt', 'neglbs', 'reflb'] if k in task_extparms))
    elif task_type == 'sentsim':
        ds_kwargs.update({'ynormfunc':task_extparms.setdefault('ynormfunc', None)})
    global_all_binlb = {}

    ext_params = dict([(k, getattr(opts, k)) if hasattr(opts, k) else (k, v) for k, v in config.clf_ext_params.items()])
    if hasattr(config, 'embed_type') and config.embed_type: ext_params['embed_type'] = config.embed_type
    task_params = dict([(k, getattr(opts, k)) if hasattr(opts, k) and getattr(opts, k) is not None else (k, v) for k, v in task_extparms.setdefault('mdlcfg', {}).items()])
    print('Classifier hyper-parameters: %s' % ext_params)
    print('Classifier task-related parameters: %s' % task_params)
    orig_epochs = mltclf_epochs = opts.epochs
    elapsed_mltclf_epochs, opts.epochs = 0, 1
    if (opts.resume):
        # Load model
        clf, prv_optimizer, resume, chckpnt = load_model(opts.resume)
        if opts.refresh:
            print('Refreshing and saving the model with newest code...')
            try:
                save_model(clf, prv_optimizer, '%s_%s.pth' % (opts.task, opts.model))
            except Exception as e:
                print(e)
        elapsed_mltclf_epochs, all_binlb = chckpnt.setdefault('mltclf_epochs', 0), clf.binlb
        # Update parameters
        clf.update_params(task_params=task_params, **ext_params)
        if (use_gpu): clf = _handle_model(clf, dev_id=dev_id, distrb=opts.distrb)
        # optmzr_cls = OPTMZR_MAP.setdefault(opts.model.split('_')[0], (torch.optim.Adam, {}, None))
        optmzr_cls = config.optmzr if config.optmzr else (torch.optim.Adam, {}, None)
        optimizer = optmzr_cls[0](clf.parameters(), lr=opts.lr, weight_decay=opts.wdecay, **optmzr_cls[1]) if opts.optim == 'adam' else torch.optim.SGD(clf.parameters(), lr=opts.lr, momentum=0.9)
        if prv_optimizer: optimizer.load_state_dict(prv_optimizer.state_dict())
        training_steps = int(len(train_ds) / opts.bsize) if hasattr(train_ds, '__len__') else opts.trainsteps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opts.wrmprop, num_training_steps=training_steps) if not opts.noschdlr and len(optmzr_cls) > 2 and optmzr_cls[2] and optmzr_cls[2] == 'linwarm' else None
        print((optimizer, scheduler))
    else:
        # Build model
        lm_model = gen_mdl(mdl_name, config, pretrained=True if type(opts.pretrained) is str and opts.pretrained.lower() == 'true' else opts.pretrained, use_gpu=use_gpu, distrb=opts.distrb, dev_id=dev_id) if mdl_name != 'none' else None
        clf = gen_clf(opts.model, config, opts.encoder, lm_model=lm_model, constraints=opts.cnstrnts.split(',') if opts.cnstrnts else [], task_type=task_type, mlt_trnsfmr=True if task_type in ['entlmnt', 'sentsim'] and task_params.setdefault('sentsim_func', None) is not None else False, task_params=task_params, use_gpu=use_gpu, distrb=opts.distrb, dev_id=dev_id, **ext_params)
        # optmzr_cls = OPTMZR_MAP.setdefault(opts.model.split('_')[0], (torch.optim.Adam, {}, None))
        optmzr_cls = config.optmzr if config.optmzr else (torch.optim.Adam, {}, None)
        optimizer = optmzr_cls[0](clf.parameters(), lr=opts.lr, weight_decay=opts.wdecay, **optmzr_cls[1]) if opts.optim == 'adam' else torch.optim.SGD(clf.parameters(), lr=opts.lr, momentum=0.9)
        training_steps = int(len(train_ds) / opts.bsize) if hasattr(train_ds, '__len__') else opts.trainsteps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opts.wrmprop, num_training_steps=training_steps) if not opts.noschdlr and len(optmzr_cls) > 2 and optmzr_cls[2] and optmzr_cls[2] == 'linwarm' else None
        print((optimizer, scheduler))

    # Prepare data
    print('Dataset path: %s' % os.path.join(DATA_PATH, task_path))
    num_clfs = min([len(fs.listf(os.path.join(DATA_PATH, task_path), pattern='%s_\d.csv' % x)) for x in ['train', 'dev', 'test']])
    for epoch in range(elapsed_mltclf_epochs, mltclf_epochs):
        print('Global %i epoch(s)...' % epoch)
        clf.reset_global_binlb()
        all_binlb = {}
        for i in range(num_clfs):
            print('Training on the %s sub-dataset...' % iflteng.ordinal(i+1))
            train_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'train_%i.%s' % (i, opts.fmt)), task_cols['X'], task_cols['y'], config.encode_func, tokenizer, config, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms else None, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
            new_lbs = [k for k in train_ds.binlb.keys() if k not in all_binlb]
            all_binlb.update(dict([(k, v) for k, v in zip(new_lbs, range(len(all_binlb), len(all_binlb)+len(new_lbs)))]))
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
                class_weights *= (opts.clswfac[min(len(opts.clswfac)-1, i)] if type(opts.clswfac) is list else opts.clswfac)
                sampler = WeightedRandomSampler(weights=class_weights, num_samples=opts.bsize, replacement=True)
                if type(dev_id) is list: class_weights = class_weights.repeat(len(dev_id))
            train_loader = DataLoader(train_ds, batch_size=opts.bsize, shuffle=False, sampler=None, num_workers=opts.np, drop_last=opts.droplast)

            dev_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'dev_%i.%s' % (i, opts.fmt)), task_cols['X'], task_cols['y'], config.encode_func, tokenizer, config, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else all_binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
            if mdl_name.startswith('bert'): dev_ds = MaskedLMIterDataset(train_ds) if isinstance(train_ds, BaseIterDataset) else MaskedLMDataset(dev_ds)
            dev_loader = DataLoader(dev_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)
            test_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'test_%i.%s' % (i, opts.fmt)), task_cols['X'], task_cols['y'], config.encode_func, tokenizer, config, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else all_binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
            if mdl_name.startswith('bert'): test_ds = MaskedLMIterDataset(train_ds) if isinstance(train_ds, BaseIterDataset) else MaskedLMDataset(test_ds)
            test_loader = DataLoader(test_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)
            # print((len(train_ds.binlb), len(dev_ds.binlb), len(test_ds.binlb)))

            # Adjust the model
            clf.get_linear(binlb=train_ds.binlb, idx=i)

            # Training on splitted datasets
            train(clf, optimizer, train_loader, config, special_tknids_args, scheduler=scheduler, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), weights=class_weights, lmcoef=opts.lmcoef, clipmaxn=opts.clipmaxn, epochs=opts.epochs, earlystop=opts.earlystop, earlystop_delta=opts.es_delta, earlystop_patience=opts.es_patience, task_type=task_type, task_name=opts.task, mdl_name=opts.model, use_gpu=use_gpu, devq=dev_id, resume=resume if opts.resume else {}, chckpnt_kwargs=dict(mltclf_epochs=epoch))

            # Adjust the model
            clf_trnsfmr = MultiClfTransformer(clf)
            clf_trnsfmr.merge_linear(num_linear=i+1)
            clf.linear = _handle_model(clf.linear, dev_id=dev_id, distrb=opts.distrb)

            # Evaluating on the accumulated dev and test sets
            eval(clf, dev_loader, config, dev_ds.binlbr, special_tknids_args, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='dev', mdl_name=opts.model, use_gpu=use_gpu, ignored_label=task_extparms.setdefault('ignored_label', None))
            eval(clf, test_loader, config, test_ds.binlbr, special_tknids_args, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='test', mdl_name=opts.model, use_gpu=use_gpu, ignored_label=task_extparms.setdefault('ignored_label', None))
        global_all_binlb.update(all_binlb)
        # clf.binlb = all_binlb
        # clf.binlbr = dict([(v, k) for k, v in all_binlb.items()])
    else:
        if orig_epochs > 0:
            try:
                save_model(clf, optimizer, '%s_%s.pth' % (opts.task, opts.model), devq=dev_id, distrb=opts.distrb)
            except Exception as e:
                print(e)
    opts.epochs = orig_epochs

    if opts.noeval: return
    dev_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'dev.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer, config, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else all_binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): dev_ds = MaskedLMIterDataset(train_ds) if isinstance(train_ds, BaseIterDataset) else MaskedLMDataset(dev_ds)
    dev_loader = DataLoader(dev_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)
    test_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'test.%s' % opts.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer, config, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else all_binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): test_ds = MaskedLMIterDataset(train_ds) if isinstance(train_ds, BaseIterDataset) else MaskedLMDataset(test_ds)
    test_loader = DataLoader(test_ds, batch_size=opts.bsize, shuffle=False, num_workers=opts.np)

    # Evaluation
    eval(clf, dev_loader, config, dev_ds.binlbr, special_tknids_args, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='dev', mdl_name=opts.model, use_gpu=use_gpu, ignored_label=task_extparms.setdefault('ignored_label', None))
    if opts.traindev: train(clf, optimizer, dev_loader, config, special_tknids_args, scheduler=scheduler, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), weights=class_weights, lmcoef=opts.lmcoef, clipmaxn=opts.clipmaxn, epochs=orig_epochs, earlystop=opts.earlystop, earlystop_delta=opts.es_delta, earlystop_patience=opts.es_patience, task_type=task_type, task_name=opts.task, mdl_name=opts.model, use_gpu=use_gpu, devq=dev_id)
    eval(clf, test_loader, config, test_ds.binlbr, special_tknids_args, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=opts.task, ds_name='test', mdl_name=opts.model, use_gpu=use_gpu, ignored_label=task_extparms.setdefault('ignored_label', None))


def main():
    # test_config()
    # return
    if any(opts.task == t for t in ['bc5cdr-chem', 'bc5cdr-dz', 'shareclefe', 'copdner', 'ddi', 'chemprot', 'i2b2', 'hoc', 'biolarkgsc', 'copd', 'phenopubs', 'meshpubs', 'meshpubs_entilement', 'bioasqa', 'phenochf', 'toxic', 'mednli', 'snli', 'mnli', 'wnli', 'qnli', 'biolarkgsc_entilement', 'copd_entilement', 'hpo_entilement', 'biosses', 'clnclsts', 'cncpt-ddi']):
        if (opts.method == 'classify'):
            main_func = classify
        elif (opts.method == 'mltphz'):
            main_func = mltphz_clf
        elif (opts.method == 'multiclf'):
            main_func = multi_clf
        elif (opts.method == 'simrank'):
            main_func = siamese_rank
        elif (opts.method == 'simsearch_smsrerank'):
            main_func = simsearch_smsrerank
        elif (opts.method == 'simsearch'):
            main_func = simsearch
        elif (opts.method == 'rerank'):
            main_func = rerank
    else:
        return
    # Update config
    cfg_kwargs = {} if opts.cfg is None else ast.literal_eval(opts.cfg)
    opts.cfg = cfg_kwargs
    _update_cfgs(cfg_kwargs)
    global cfgr
    cfgr = io.cfg_reader(CONFIG_FILE)

    if (opts.distrb):
        global DATA_PATH
        DATA_PATH = os.path.join('/', 'data', 'bionlp')
        torch.cuda.set_device(hvd.local_rank())
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
    op.add_option('-f', '--fmt', default='tsv', help='data stored format: tsv or csv [default: %default]')
    # op.add_option('-s', '--spfmt', default='csr', help='sparse data stored format: csc or csr [default: %default]')
    # op.add_option('-a', '--avg', default='micro', help='averaging strategy for performance metrics: micro or macro [default: %default]')
    op.add_option('-j', '--epochs', default=1, action='store', type='int', dest='epochs', help='indicate the epoch used in deep learning')
    op.add_option('-z', '--bsize', default=64, action='store', type='int', dest='bsize', help='indicate the batch size used in deep learning')
    op.add_option('-o', '--omp', default=False, action='store_true', dest='omp', help='use openmp multi-thread')
    op.add_option('-g', '--gpunum', default=1, action='store', type='int', dest='gpunum', help='indicate the gpu device number')
    op.add_option('-q', '--gpuq', dest='gpuq', help='prefered gpu device queue [template: DEVICE_ID1,DEVICE_ID2,...,DEVICE_IDn]')
    op.add_option('--gpumem', default=0.5, action='store', type='float', dest='gpumem', help='indicate the per process gpu memory fraction')
    op.add_option('--crsdev', default=False, action='store_true', dest='crsdev', help='whether to use heterogeneous devices')
    op.add_option('--distrb', default=False, action='store_true', dest='distrb', help='whether to distribute data over multiple devices')
    op.add_option('--distbknd', default='nccl', action='store', dest='distbknd', help='distribute framework backend')
    op.add_option('--disturl', default='env://', action='store', dest='disturl', help='distribute framework url')
    op.add_option('--optim', default='adam', action='store', dest='optim', help='indicate the optimizer')
    op.add_option('--wrmprop', default=0.1, action='store', type='float', dest='wrmprop', help='indicate the warmup proportion')
    op.add_option('--trainsteps', default=1000, action='store', type='int', dest='trainsteps', help='indicate the training steps')
    op.add_option('--traindev', default=False, action='store_true', dest='traindev', help='whether to use dev dataset for training')
    op.add_option('--noeval', default=False, action='store_true', dest='noeval', help='whether to train only')
    op.add_option('--noschdlr', default=False, action='store_true', dest='noschdlr', help='force to not use scheduler whatever the default setting is')
    op.add_option('--earlystop', default=False, action='store_true', dest='earlystop', help='whether to use early stopping')
    op.add_option('--es_patience', default=5, action='store', type='int', dest='es_patience', help='indicate the tolerance time for training metric violation')
    op.add_option('--es_delta', default=float(5e-3), action='store', type='float', dest='es_delta', help='indicate the minimum delta of early stopping')
    op.add_option('--vocab', dest='vocab', help='vocabulary file')
    op.add_option('--bpe', dest='bpe', help='bpe merge file')
    op.add_option('--w2v', dest='w2v_path', help='word2vec model file')
    op.add_option('--sentvec', dest='sentvec_path', help='sentvec model file')
    op.add_option('--maxlen', default=128, action='store', type='int', dest='maxlen', help='indicate the maximum sequence length for each samples')
    op.add_option('--maxtrial', default=50, action='store', type='int', dest='maxtrial', help='maximum time to try')
    op.add_option('--initln', default=False, action='store_true', dest='initln', help='whether to initialize the linear layer')
    op.add_option('--initln_mean', default=0., action='store', type='float', dest='initln_mean', help='indicate the mean of the parameters in linear model when Initializing')
    op.add_option('--initln_std', default=0.02, action='store', type='float', dest='initln_std', help='indicate the standard deviation of the parameters in linear model when Initializing')
    op.add_option('--weight_class', default=False, action='store_true', dest='weight_class', help='whether to drop the last incompleted batch')
    op.add_option('--clswfac', default='1', action='store', type='str', dest='clswfac', help='whether to drop the last incompleted batch')
    op.add_option('--droplast', default=False, action='store_true', dest='droplast', help='whether to drop the last incompleted batch')
    op.add_option('--do_norm', default=False, action='store_true', dest='do_norm', help='whether to do normalization')
    op.add_option('--norm_type', default='batch', action='store', dest='norm_type', help='normalization layer class')
    op.add_option('--do_extlin', default=False, action='store_true', dest='do_extlin', help='whether to apply additional fully-connected layer to the hidden states of the language model')
    op.add_option('--do_lastdrop', default=False, action='store_true', dest='do_lastdrop', help='whether to apply dropout to the last layer')
    op.add_option('--lm_loss', default=False, action='store_true', dest='lm_loss', help='whether to apply dropout to the last layer')
    op.add_option('--do_crf', default=False, action='store_true', dest='do_crf', help='whether to apply CRF layer')
    op.add_option('--do_thrshld', default=False, action='store_true', dest='do_thrshld', help='whether to apply ThresholdEstimator layer')
    op.add_option('--fchdim', default=0, action='store', type='int', dest='fchdim', help='indicate the dimensions of the hidden layers in the Embedding-based classifier, 0 means using only one linear layer')
    op.add_option('--iactvtn', default='relu', action='store', dest='iactvtn', help='indicate the internal activation function')
    op.add_option('--oactvtn', default='sigmoid', action='store', dest='oactvtn', help='indicate the output activation function')
    op.add_option('--bert_outlayer', default='-1', action='store', type='str', dest='output_layer', help='indicate which layer to be the output of BERT model')
    op.add_option('--pooler', dest='pooler', help='indicate the pooling strategy when selecting features: max or avg')
    op.add_option('--seq2seq', dest='seq2seq', help='indicate the seq2seq strategy when converting sequences of embeddings into a vector')
    op.add_option('--seq2vec', dest='seq2vec', help='indicate the seq2vec strategy when converting sequences of embeddings into a vector: pytorch-lstm, cnn, or cnn_highway')
    op.add_option('--ssfunc', dest='sentsim_func', help='indicate the sentence similarity metric [dist|sim]')
    op.add_option('--catform', dest='concat_strategy', help='indicate the sentence similarity metric [normal|diff]')
    op.add_option('--ymode', default='sim', dest='ymode', help='indicate the sentence similarity metric in gold standard [dist|sim]')
    op.add_option('--loss', dest='loss', help='indicate the loss function')
    op.add_option('--cnstrnts', dest='cnstrnts', help='indicate the constraint scheme')
    op.add_option('--lr', default=float(1e-3), action='store', type='float', dest='lr', help='indicate the learning rate of the optimizer')
    op.add_option('--wdecay', default=float(1e-5), action='store', type='float', dest='wdecay', help='indicate the weight decay of the optimizer')
    op.add_option('--lmcoef', default=0.5, action='store', type='float', dest='lmcoef', help='indicate the coefficient of the language model loss when fine tuning')
    op.add_option('--sampfrac', action='store', type='float', dest='sampfrac', help='indicate the sampling fraction for datasets')
    op.add_option('--pdrop', default=0.2, action='store', type='float', dest='pdrop', help='indicate the dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler')
    op.add_option('--pthrshld', default=0.5, action='store', type='float', dest='pthrshld', help='indicate the threshold for predictive probabilitiy')
    op.add_option('--topk', default=5, action='store', type='int', dest='topk', help='indicate the top k search parameter')
    op.add_option('--do_tfidf', default=False, action='store_true', dest='do_tfidf', help='whether to use tfidf as text features')
    op.add_option('--do_chartfidf', default=False, action='store_true', dest='do_chartfidf', help='whether to use charater tfidf as text features')
    op.add_option('--do_bm25', default=False, action='store_true', dest='do_bm25', help='whether to use bm25 as text features')
    op.add_option('--clipmaxn', action='store', type='float', dest='clipmaxn', help='indicate the max norm of the gradients')
    op.add_option('--resume', action='store', dest='resume', help='resume training model file')
    op.add_option('--refresh', default=False, action='store_true', dest='refresh', help='refresh the trained model with newest code')
    op.add_option('--sampw', default=False, action='store_true', dest='sample_weights', help='use sample weights')
    op.add_option('--corpus', help='corpus data')
    op.add_option('--onto', help='ontology data')
    op.add_option('--pred', help='prediction file')
    op.add_option('-i', '--input', help='input dataset')
    op.add_option('-w', '--cache', default='.cache', help='the location of cache files')
    op.add_option('-u', '--task', default='ddi', type='str', dest='task', help='the task name [default: %default]')
    op.add_option('--model', default='gpt2', type='str', dest='model', help='the model to be validated')
    op.add_option('--encoder', dest='encoder', help='the encoder to be used after the language model: pool, s2v or s2s')
    op.add_option('--pretrained', dest='pretrained', help='pretrained model file')
    op.add_option('--seed', dest='seed', help='manually set the random seed')
    op.add_option('-c', '--cfg', help='config string used to update the settings, format: {\'param_name1\':param_value1[, \'param_name1\':param_value1]}')
    op.add_option('-m', '--method', default='classify', help='main method to run')
    op.add_option('-v', '--verbose', default=False, action='store_true', dest='verbose', help='display detailed information')

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
        if opts.verbose: os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[:opts.gpunum]) if 'CUDA_VISIBLE_DEVICES' in os.environ and not os.environ['CUDA_VISIBLE_DEVICES'].isspace() else ','.join(map(str, opts.gpuq[:opts.gpunum]))
        setattr(opts, 'devq', list(range(torch.cuda.device_count())))
    else:
        setattr(opts, 'devq', None)
    if opts.distrb:
        import horovod.torch as hvd
        hvd.init()

    if opts.seed is not None:
        np.random.seed(opts.seed)
        torch.cuda.manual_seed(opts.seed)

    opts.output_layer = list(map(int, opts.output_layer.split(',')))
    opts.output_layer = opts.output_layer[0] if len(opts.output_layer) == 1 else opts.output_layer
    opts.clswfac = list(map(float, opts.clswfac.split(','))) if ',' in opts.clswfac else float(opts.clswfac)

    main()
