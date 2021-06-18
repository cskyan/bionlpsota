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
import argparse

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

global CONFIG_FILE, DATA_PATH, SC, cfgr, args
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
args, cfgr = {}, None


def classify(dev_id=None):
    # Prepare model related meta data
    mdl_name = args.model.lower().replace(' ', '_')
    common_cfg = cfgr('validate', 'common')
    pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
    config_kwargs = dict([(k, v) for k, v in args.__dict__.items() if not k.startswith('_') and k not in set(['dataset', 'model', 'template']) and v is not None and not callable(v)])
    config = Configurable(args.task, mdl_name, common_cfg=common_cfg, wsdir=PAR_DIR, **config_kwargs)
    params = pr('LM', config.lm_params) if mdl_name != 'none' else {}
    use_gpu = dev_id is not None
    tokenizer = config.tknzr.from_pretrained(params['pretrained_vocab_path'] if 'pretrained_vocab_path' in params else config.lm_mdl_name) if config.tknzr else {}
    _adjust_encoder(tokenizer, config)

    # Prepare task related meta data.
    task_path, task_type, task_dstype, task_cols, task_trsfm, task_extparms = config.input if config.input and os.path.isdir(os.path.join(DATA_PATH, config.input)) else config.task_path, config.task_type, config.task_ds, config.task_col, config.task_trsfm, config.task_ext_params
    ds_kwargs = config.ds_kwargs

    # Prepare data
    if (not config.distrb or config.distrb and hvd.rank() == 0): logging.info('Dataset path: %s' % os.path.join(DATA_PATH, task_path))
    train_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'train.%s' % config.fmt), tokenizer, config, **ds_kwargs)
    # Calculate the class weights if needed
    lb_trsfm = [x['get_lb'] for x in task_trsfm[1] if 'get_lb' in x]
    if (not config.weight_class or task_type == 'sentsim'):
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
        sampler = WeightedRandomSampler(weights=class_weights, num_samples=config.bsize, replacement=True)
        if not config.distrb and type(dev_id) is list: class_weights = class_weights.repeat(len(dev_id))

    # Partition dataset among workers using DistributedSampler
    if config.distrb: sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=hvd.size(), rank=hvd.rank())

    train_loader = DataLoader(train_ds, batch_size=config.bsize, shuffle=sampler is None and config.droplast, sampler=sampler, num_workers=config.np, drop_last=config.droplast)

    # Classifier
    if (not config.distrb or config.distrb and hvd.rank() == 0):
        logging.info('Language model input fields: %s' % config.input_keys)
        logging.info('Classifier hyper-parameters: %s' % config.clf_ext_params)
        logging.info('Classifier task-related parameters: %s' % task_extparms['mdlaware'])
    if (config.resume):
        # Load model
        clf, prv_optimizer, resume, chckpnt = load_model(config.resume)
        if config.refresh:
            logging.info('Refreshing and saving the model with newest code...')
            try:
                if (not distrb or distrb and hvd.rank() == 0):
                    save_model(clf, prv_optimizer, '%s_%s.pth' % (config.task, config.model))
            except Exception as e:
                logging.warning(e)
        # Update parameters
        clf.update_params(task_params=task_extparms['mdlaware'], **config.clf_ext_params)
        if (use_gpu): clf = _handle_model(clf, dev_id=dev_id, distrb=config.distrb)
        # Construct optimizer
        optmzr_cls = config.optmzr if config.optmzr else (torch.optim.Adam, {}, None)
        optimizer = optmzr_cls[0](clf.parameters(), lr=config.lr, weight_decay=config.wdecay, **optmzr_cls[1]) if config.optim == 'adam' else torch.optim.SGD(clf.parameters(), lr=config.lr, momentum=0.9)
        if prv_optimizer: optimizer.load_state_dict(prv_optimizer.state_dict())
        training_steps = int(len(train_ds) / config.bsize) if hasattr(train_ds, '__len__') else config.trainsteps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(config.wrmprop*training_steps), num_training_steps=training_steps) if not config.noschdlr and len(optmzr_cls) > 2 and optmzr_cls[2] and optmzr_cls[2] == 'linwarm' else None
        if (not config.distrb or config.distrb and hvd.rank() == 0): logging.info((optimizer, scheduler))
    else:
        # Build model
        lm_model, lm_config = gen_mdl(config, use_gpu=use_gpu, distrb=config.distrb, dev_id=dev_id)
        clf = gen_clf(config, lm_model, lm_config, num_lbs=len(train_ds.binlb) if train_ds.binlb else 1, mlt_trnsfmr=True if task_type in ['entlmnt', 'sentsim'] and task_extparms['mdlaware'].setdefault('sentsim_func', None) is not None else False, task_params=task_extparms['mdlaware'], binlb=train_ds.binlb, binlbr=train_ds.binlbr, use_gpu=use_gpu, distrb=config.distrb, dev_id=dev_id, **config.clf_ext_params)
        optmzr_cls = config.optmzr if config.optmzr else (torch.optim.Adam, {}, None)
        optimizer = optmzr_cls[0](clf.parameters(), lr=config.lr, weight_decay=config.wdecay, **optmzr_cls[1]) if config.optim == 'adam' else torch.optim.SGD(clf.parameters(), lr=config.lr, momentum=0.9)
        training_steps = int(len(train_ds) / config.bsize) if hasattr(train_ds, '__len__') else config.trainsteps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.wrmprop, num_training_steps=training_steps) if not config.noschdlr and len(optmzr_cls) > 2 and optmzr_cls[2] and optmzr_cls[2] == 'linwarm' else None
        if (not config.distrb or config.distrb and hvd.rank() == 0): logging.info((optimizer, scheduler))

    config.execute_all_callback()
    if config.verbose: logging.debug(config.__dict__)
    if config.configfmt == 'yaml':
        config.to_yaml()
    else:
        config.to_json()

    if config.distrb:
        # Add Horovod Distributed Optimizer
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=clf.named_parameters())
        # Broadcast parameters from rank 0 to all other processes.
        hvd.broadcast_parameters(clf.state_dict(), root_rank=0)

    # Training
    train(clf, optimizer, train_loader, config, scheduler, weights=class_weights, lmcoef=config.lmcoef, clipmaxn=config.clipmaxn, epochs=config.epochs, earlystop=config.earlystop, earlystop_delta=config.es_delta, earlystop_patience=config.es_patience, use_gpu=use_gpu, devq=dev_id, distrb=config.distrb, resume=resume if config.resume else {})

    if config.distrb:
        if hvd.rank() == 0:
            clf = _handle_model(clf, dev_id=dev_id, distrb=False)
        else:
            return

    if config.noeval: return
    dev_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'dev.%s' % config.fmt), tokenizer, config, binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else train_ds.binlb, **ds_kwargs)
    dev_loader = DataLoader(dev_ds, batch_size=config.bsize, shuffle=False, num_workers=config.np)
    test_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'test.%s' % config.fmt), tokenizer, config, binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else train_ds.binlb, **ds_kwargs)
    test_loader = DataLoader(test_ds, batch_size=config.bsize, shuffle=False, num_workers=config.np)
    logging.debug(('binlb', train_ds.binlb, dev_ds.binlb, test_ds.binlb))

    # Evaluation
    eval(clf, dev_loader, config, ds_name='dev', use_gpu=use_gpu, devq=dev_id, distrb=config.distrb, ignored_label=task_extparms.setdefault('ignored_label', None))
    if config.traindev: train(clf, optimizer, dev_loader, config, scheduler=scheduler, weights=class_weights, lmcoef=config.lmcoef, clipmaxn=config.clipmaxn, epochs=config.epochs, earlystop=config.earlystop, earlystop_delta=config.es_delta, earlystop_patience=config.es_patience, use_gpu=use_gpu, devq=dev_id, distrb=config.distrb)
    eval(clf, test_loader, config, ds_name='test', use_gpu=use_gpu, devq=dev_id, distrb=config.distrb, ignored_label=task_extparms.setdefault('ignored_label', None))


def mltphz_clf(dev_id=None):
    cfg_kwargs = args.cfg
    common_cfg_kwargs = dict([(k, v) for k, v in cfg_kwargs.items() if k != 'phases'])
    _update_cfgs(common_cfg_kwargs)
    for cfgs in cfg_kwargs['phases']:
        _update_cfgs(cfgs)
        classify(dev_id=dev_id)
        setattr(args, 'resume', '%s_%s.pth' % (args.task, args.model))


def multi_clf(dev_id=None):
    '''Train multiple classifiers and use them to predict multiple set of labels'''
    import inflect
    from bionlp.util import fs
    iflteng = inflect.engine()

    logging.info('### Multi Classifier Head Mode ###')
    # Prepare model related meta data
    mdl_name = args.model.lower().replace(' ', '_')
    common_cfg = cfgr('validate', 'common')
    pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
    config_kwargs = dict([(k, v) for k, v in args.__dict__.items() if not k.startswith('_') and k not in set(['dataset', 'model', 'template']) and v is not None and type(v) is not function])
    config = Configurable(args.task, mdl_name, common_cfg=common_cfg, wsdir=PAR_DIR, **config_kwargs)
    params = pr('LM', config.lm_params) if mdl_name != 'none' else {}
    use_gpu = dev_id is not None
    tokenizer = config.tknzr.from_pretrained(params['pretrained_vocab_path'] if 'pretrained_vocab_path' in params else config.lm_mdl_name) if config.tknzr else None
    task_type = config.task_type
    _adjust_encoder(tokenizer, config)
    special_tknids_args = dict(zip(special_tkns[0], special_tknids))
    task_trsfm_kwargs = dict(list(zip(special_tkns[0], special_tknids))+[('model',args.model), ('sentsim_func', args.sentsim_func), ('seqlen',args.maxlen)])
    # Prepare task related meta data.
    task_path, task_dstype, task_cols, task_trsfm, task_extparms = args.input if args.input and os.path.isdir(os.path.join(DATA_PATH, args.input)) else config.task_path, config.task_ds, config.task_col, config.task_trsfm, config.task_ext_params
    trsfms = (task_trsfm[0] if len(task_trsfm) > 0 else [])
    # trsfms_kwargs = ([] if args.model in LM_EMBED_MDL_MAP else ([{'seqlen':args.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}] if TASK_TYPE_MAP[args.task]=='nmt' else [{'seqlen':args.maxlen, 'trimlbs':task_extparms.setdefault('trimlbs', False), 'special_tkns':special_tknids_args}, task_trsfm_kwargs, {'seqlen':args.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}])) + (task_trsfm[1] if len(task_trsfm) >= 2 else [{}] * len(task_trsfm[0]))
    trsfms_kwargs = ([] if hasattr(config, 'embed_type') and config.embed_type else ([{'seqlen':args.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}] if config.task_type=='nmt' else [{'seqlen':args.maxlen, 'trimlbs':task_extparms.setdefault('trimlbs', False), 'required_special_tkns':['start_tknids', 'clf_tknids', 'delim_tknids'] if task_type in ['entlmnt', 'sentsim'] and (task_extparms.setdefault('sentsim_func', None) is None or not mdl_name.startswith('bert')) else ['start_tknids', 'clf_tknids'], 'special_tkns':special_tknids_args}, task_trsfm_kwargs, {'seqlen':args.maxlen, 'xpad_val':task_extparms.setdefault('xpad_val', 0), 'ypad_val':task_extparms.setdefault('ypad_val', None)}])) + (task_trsfm[1] if len(task_trsfm) >= 2 else [{}] * len(task_trsfm[0]))
    ds_kwargs = {'sampw':args.sample_weights, 'sampfrac':args.sampfrac}
    if task_type == 'nmt':
        ds_kwargs.update({'lb_coding':task_extparms.setdefault('lb_coding', 'IOB')})
    elif task_type == 'entlmnt':
        ds_kwargs.update(dict((k, task_extparms[k]) for k in ['origlb', 'lbtxt', 'neglbs', 'reflb'] if k in task_extparms))
    elif task_type == 'sentsim':
        ds_kwargs.update({'ynormfunc':task_extparms.setdefault('ynormfunc', None)})
    global_all_binlb = {}

    ext_params = dict([(k, getattr(args, k)) if hasattr(args, k) else (k, v) for k, v in config.clf_ext_params.items()])
    if hasattr(config, 'embed_type') and config.embed_type: ext_params['embed_type'] = config.embed_type
    task_params = dict([(k, getattr(args, k)) if hasattr(args, k) and getattr(args, k) is not None else (k, v) for k, v in task_extparms.setdefault('mdlcfg', {}).items()])
    logging.info('Classifier hyper-parameters: %s' % ext_params)
    logging.info('Classifier task-related parameters: %s' % task_params)
    orig_epochs = mltclf_epochs = args.epochs
    elapsed_mltclf_epochs, args.epochs = 0, 1
    if (args.resume):
        # Load model
        clf, prv_optimizer, resume, chckpnt = load_model(args.resume)
        if args.refresh:
            logging.info('Refreshing and saving the model with newest code...')
            try:
                save_model(clf, prv_optimizer, '%s_%s.pth' % (args.task, args.model))
            except Exception as e:
                logging.warning(e)
        elapsed_mltclf_epochs, all_binlb = chckpnt.setdefault('mltclf_epochs', 0), clf.binlb
        # Update parameters
        clf.update_params(task_params=task_params, **ext_params)
        if (use_gpu): clf = _handle_model(clf, dev_id=dev_id, distrb=args.distrb)
        # optmzr_cls = OPTMZR_MAP.setdefault(args.model.split('_')[0], (torch.optim.Adam, {}, None))
        optmzr_cls = config.optmzr if config.optmzr else (torch.optim.Adam, {}, None)
        optimizer = optmzr_cls[0](clf.parameters(), lr=args.lr, weight_decay=args.wdecay, **optmzr_cls[1]) if args.optim == 'adam' else torch.optim.SGD(clf.parameters(), lr=args.lr, momentum=0.9)
        if prv_optimizer: optimizer.load_state_dict(prv_optimizer.state_dict())
        training_steps = int(len(train_ds) / args.bsize) if hasattr(train_ds, '__len__') else args.trainsteps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.wrmprop, num_training_steps=training_steps) if not args.noschdlr and len(optmzr_cls) > 2 and optmzr_cls[2] and optmzr_cls[2] == 'linwarm' else None
        logging.info((optimizer, scheduler))
    else:
        # Build model
        lm_model = gen_mdl(mdl_name, config, pretrained=True if type(args.pretrained) is str and args.pretrained.lower() == 'true' else args.pretrained, use_gpu=use_gpu, distrb=args.distrb, dev_id=dev_id) if mdl_name != 'none' else None
        clf = gen_clf(args.model, config, args.encoder, lm_model=lm_model, mlt_trnsfmr=True if task_type in ['entlmnt', 'sentsim'] and task_params.setdefault('sentsim_func', None) is not None else False, task_params=task_params, use_gpu=use_gpu, distrb=args.distrb, dev_id=dev_id, **ext_params)
        # optmzr_cls = OPTMZR_MAP.setdefault(args.model.split('_')[0], (torch.optim.Adam, {}, None))
        optmzr_cls = config.optmzr if config.optmzr else (torch.optim.Adam, {}, None)
        optimizer = optmzr_cls[0](clf.parameters(), lr=args.lr, weight_decay=args.wdecay, **optmzr_cls[1]) if args.optim == 'adam' else torch.optim.SGD(clf.parameters(), lr=args.lr, momentum=0.9)
        training_steps = int(len(train_ds) / args.bsize) if hasattr(train_ds, '__len__') else args.trainsteps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.wrmprop, num_training_steps=training_steps) if not args.noschdlr and len(optmzr_cls) > 2 and optmzr_cls[2] and optmzr_cls[2] == 'linwarm' else None
        logging.info((optimizer, scheduler))

    # Prepare data
    logging.info('Dataset path: %s' % os.path.join(DATA_PATH, task_path))
    num_clfs = min([len(fs.listf(os.path.join(DATA_PATH, task_path), pattern='%s_\d.csv' % x)) for x in ['train', 'dev', 'test']])
    for epoch in range(elapsed_mltclf_epochs, mltclf_epochs):
        logging.info('Global %i epoch(s)...' % epoch)
        clf.reset_global_binlb()
        all_binlb = {}
        for i in range(num_clfs):
            logging.info('Training on the %s sub-dataset...' % iflteng.ordinal(i+1))
            train_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'train_%i.%s' % (i, args.fmt)), task_cols['X'], task_cols['y'], config.encode_func, tokenizer, config, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms else None, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
            new_lbs = [k for k in train_ds.binlb.keys() if k not in all_binlb]
            all_binlb.update(dict([(k, v) for k, v in zip(new_lbs, range(len(all_binlb), len(all_binlb)+len(new_lbs)))]))
            if mdl_name.startswith('bert'): train_ds = MaskedLMIterDataset(train_ds) if isinstance(train_ds, BaseIterDataset) else MaskedLMDataset(train_ds)
            lb_trsfm = [x['get_lb'] for x in task_trsfm[1] if 'get_lb' in x]
            if (not args.weight_class or task_type == 'sentsim'):
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
                class_weights *= (args.clswfac[min(len(args.clswfac)-1, i)] if type(args.clswfac) is list else args.clswfac)
                sampler = WeightedRandomSampler(weights=class_weights, num_samples=args.bsize, replacement=True)
                if type(dev_id) is list: class_weights = class_weights.repeat(len(dev_id))
            train_loader = DataLoader(train_ds, batch_size=args.bsize, shuffle=False, sampler=None, num_workers=args.np, drop_last=args.droplast)

            dev_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'dev_%i.%s' % (i, args.fmt)), task_cols['X'], task_cols['y'], config.encode_func, tokenizer, config, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else all_binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
            if mdl_name.startswith('bert'): dev_ds = MaskedLMIterDataset(train_ds) if isinstance(train_ds, BaseIterDataset) else MaskedLMDataset(dev_ds)
            dev_loader = DataLoader(dev_ds, batch_size=args.bsize, shuffle=False, num_workers=args.np)
            test_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'test_%i.%s' % (i, args.fmt)), task_cols['X'], task_cols['y'], config.encode_func, tokenizer, config, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else all_binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
            if mdl_name.startswith('bert'): test_ds = MaskedLMIterDataset(train_ds) if isinstance(train_ds, BaseIterDataset) else MaskedLMDataset(test_ds)
            test_loader = DataLoader(test_ds, batch_size=args.bsize, shuffle=False, num_workers=args.np)
            logging.debug(('binlb', train_ds.binlb, dev_ds.binlb, test_ds.binlb))

            # Adjust the model
            clf.get_linear(binlb=train_ds.binlb, idx=i)

            # Training on splitted datasets
            train(clf, optimizer, train_loader, config, special_tknids_args, scheduler=scheduler, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), weights=class_weights, lmcoef=args.lmcoef, clipmaxn=args.clipmaxn, epochs=args.epochs, earlystop=args.earlystop, earlystop_delta=args.es_delta, earlystop_patience=args.es_patience, task_type=task_type, task_name=args.task, mdl_name=args.model, use_gpu=use_gpu, devq=dev_id, resume=resume if args.resume else {}, chckpnt_kwargs=dict(mltclf_epochs=epoch))

            # Adjust the model
            clf_trnsfmr = MultiClfTransformer(clf)
            clf_trnsfmr.merge_linear(num_linear=i+1)
            clf.linear = _handle_model(clf.linear, dev_id=dev_id, distrb=args.distrb)

            # Evaluating on the accumulated dev and test sets
            eval(clf, dev_loader, config, dev_ds.binlbr, special_tknids_args, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=args.task, ds_name='dev', mdl_name=args.model, use_gpu=use_gpu, ignored_label=task_extparms.setdefault('ignored_label', None))
            eval(clf, test_loader, config, test_ds.binlbr, special_tknids_args, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=args.task, ds_name='test', mdl_name=args.model, use_gpu=use_gpu, ignored_label=task_extparms.setdefault('ignored_label', None))
        global_all_binlb.update(all_binlb)
        # clf.binlb = all_binlb
        # clf.binlbr = dict([(v, k) for k, v in all_binlb.items()])
    else:
        if orig_epochs > 0:
            try:
                save_model(clf, optimizer, '%s_%s.pth' % (args.task, args.model), devq=dev_id, distrb=args.distrb)
            except Exception as e:
                logging.warning(e)
    args.epochs = orig_epochs

    if args.noeval: return
    dev_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'dev.%s' % args.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer, config, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else all_binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): dev_ds = MaskedLMIterDataset(train_ds) if isinstance(train_ds, BaseIterDataset) else MaskedLMDataset(dev_ds)
    dev_loader = DataLoader(dev_ds, batch_size=args.bsize, shuffle=False, num_workers=args.np)
    test_ds = task_dstype(os.path.join(DATA_PATH, task_path, 'test.%s' % args.fmt), task_cols['X'], task_cols['y'], config.encode_func, tokenizer, config, sep='\t', index_col=task_cols['index'], binlb=task_extparms['binlb'] if 'binlb' in task_extparms and type(task_extparms['binlb']) is not str else all_binlb, transforms=trsfms, transforms_kwargs=trsfms_kwargs, mltl=task_extparms.setdefault('mltl', False), **ds_kwargs)
    if mdl_name.startswith('bert'): test_ds = MaskedLMIterDataset(train_ds) if isinstance(train_ds, BaseIterDataset) else MaskedLMDataset(test_ds)
    test_loader = DataLoader(test_ds, batch_size=args.bsize, shuffle=False, num_workers=args.np)

    # Evaluation
    eval(clf, dev_loader, config, dev_ds.binlbr, special_tknids_args, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=args.task, ds_name='dev', mdl_name=args.model, use_gpu=use_gpu, ignored_label=task_extparms.setdefault('ignored_label', None))
    if args.traindev: train(clf, optimizer, dev_loader, config, special_tknids_args, scheduler=scheduler, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), weights=class_weights, lmcoef=args.lmcoef, clipmaxn=args.clipmaxn, epochs=orig_epochs, earlystop=args.earlystop, earlystop_delta=args.es_delta, earlystop_patience=args.es_patience, task_type=task_type, task_name=args.task, mdl_name=args.model, use_gpu=use_gpu, devq=dev_id)
    eval(clf, test_loader, config, test_ds.binlbr, special_tknids_args, pad_val=(task_extparms.setdefault('xpad_val', 0), train_ds.binlb[task_extparms.setdefault('ypad_val', 0)]) if task_type=='nmt' else task_extparms.setdefault('xpad_val', 0), task_type=task_type, task_name=args.task, ds_name='test', mdl_name=args.model, use_gpu=use_gpu, ignored_label=task_extparms.setdefault('ignored_label', None))


MAIN_FUNC = {'classify':classify, 'mltphz':mltphz_clf, 'multiclf':multi_clf}


def main():
    main_func = MAIN_FUNC[args.method]

    if (args.distrb):
        main_func(args.devq if len(args.devq) > 1 else args.devq[0])
    elif (args.devq): # Single-process
        main_func(args.devq if len(args.devq) > 1 else args.devq[0])
    else:
        main_func(None) # CPU


if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser(description='Train or evaluate the model.')
    parser.add_argument('-k', '--kfold', default=10, type=int, help='indicate the K fold cross validation')
    parser.add_argument('-p', '--pid', default=0, type=int, help='indicate the process ID')
    parser.add_argument('-n', '--np', default=1, type=int, help='indicate the number of processes used for training')
    parser.add_argument('-f', '--fmt', choices=['csv', 'tsv'], default='csv', help='data stored format: tsv or csv [default: %default]')
    # parser.add_argument('-a', '--avg', default='micro', help='averaging strategy for performance metrics: micro or macro [default: %default]')
    parser.add_argument('-j', '--epochs', default=1, type=int, help='indicate the epoch used in deep learning')
    parser.add_argument('-z', '--bsize', default=64, type=int, help='indicate the batch size used in deep learning')
    parser.add_argument('-o', '--omp', default=False, action='store_true', help='use openmp multi-thread')
    parser.add_argument('-g', '--gpunum', default=1, type=int, help='indicate the gpu device number')
    parser.add_argument('-q', '--gpuq', help='prefered gpu device queue [template: DEVICE_ID1,DEVICE_ID2,...,DEVICE_IDn]')
    parser.add_argument('--gpumem', default=0.5, type=float, help='indicate the per process gpu memory fraction')
    parser.add_argument('--crsdev', default=False, action='store_true', help='whether to use heterogeneous devices')
    parser.add_argument('--distrb', default=False, action='store_true', help='whether to distribute data over multiple devices')
    parser.add_argument('--distbknd', default='nccl', help='distribute framework backend')
    parser.add_argument('--disturl', default='env://', help='distribute framework url')
    parser.add_argument('--optim', default='adam', help='indicate the optimizer')
    parser.add_argument('--wrmprop', default=0.1, type=float, help='indicate the warmup proportion')
    parser.add_argument('--trainsteps', default=1000, type=int, help='indicate the training steps')
    parser.add_argument('--traindev', default=False, action='store_true', help='whether to use dev dataset for training')
    parser.add_argument('--noeval', default=False, action='store_true', help='whether to train only')
    parser.add_argument('--noschdlr', default=False, action='store_true', help='force to not use scheduler whatever the default setting is')
    parser.add_argument('--earlystop', default=False, action='store_true', help='whether to use early stopping')
    parser.add_argument('--es_patience', default=5, type=int, help='indicate the tolerance time for training metric violation')
    parser.add_argument('--es_delta', default=float(5e-3), type=float, help='indicate the minimum delta of early stopping')
    parser.add_argument('--vocab', help='vocabulary file')
    parser.add_argument('--bpe', help='bpe merge file')
    parser.add_argument('--w2v', dest='w2v_path', help='word2vec model file')
    parser.add_argument('--sentvec', dest='sentvec_path', help='sentvec model file')
    parser.add_argument('--maxlen', default=128, type=int, help='indicate the maximum sequence length for each samples')
    parser.add_argument('--maxtrial', default=50, type=int, help='maximum time to try')
    parser.add_argument('--initln', default=False, action='store_true', help='whether to initialize the linear layer')
    parser.add_argument('--initln_mean', default=0., type=float, help='indicate the mean of the parameters in linear model when Initializing')
    parser.add_argument('--initln_std', default=0.02, type=float, help='indicate the standard deviation of the parameters in linear model when Initializing')
    parser.add_argument('--weight_class', default=False, action='store_true', help='whether to drop the last incompleted batch')
    parser.add_argument('--clswfac', default='1', type=str, help='whether to drop the last incompleted batch')
    parser.add_argument('--droplast', default=False, action='store_true', help='whether to drop the last incompleted batch')
    parser.add_argument('--do_norm', default=False, action='store_true', help='whether to do normalization')
    parser.add_argument('--norm_type', default='batch', help='normalization layer class')
    parser.add_argument('--do_extlin', default=False, action='store_true', help='whether to apply additional fully-connected layer to the hidden states of the language model')
    parser.add_argument('--do_lastdrop', default=False, action='store_true', help='whether to apply dropout to the last layer')
    parser.add_argument('--lm_loss', default=False, action='store_true', help='whether to output language model loss for optimization')
    parser.add_argument('--do_crf', default=False, action='store_true', help='whether to apply CRF layer')
    parser.add_argument('--do_thrshld', default=False, action='store_true', help='whether to apply ThresholdEstimator layer')
    parser.add_argument('--fchdim', default=0, type=int, help='indicate the dimensions of the hidden layers in the Embedding-based classifier, 0 means using only one linear layer')
    parser.add_argument('--iactvtn', default='relu', help='indicate the internal activation function')
    parser.add_argument('--oactvtn', default='sigmoid', help='indicate the output activation function')
    parser.add_argument('--bert_outlayer', default='-1', type=str, dest='output_layer', help='indicate which layer to be the output of BERT model')
    parser.add_argument('--pooler', help='indicate the pooling strategy when selecting features: max or avg')
    parser.add_argument('--seq2seq', help='indicate the seq2seq strategy when converting sequences of embeddings into a vector')
    parser.add_argument('--seq2vec', help='indicate the seq2vec strategy when converting sequences of embeddings into a vector: pytorch-lstm, cnn, or cnn_highway')
    parser.add_argument('--ssfunc', choices=['dist', 'sim'], dest='sentsim_func', help='indicate the sentence similarity metric [dist|sim]')
    parser.add_argument('--catform', choices=['normal', 'diff'], dest='concat_strategy', help='indicate the sentence similarity metric [normal|diff]')
    parser.add_argument('--ymode', default='sim', choices=['dist', 'sim'], help='indicate the sentence similarity metric in gold standard [dist|sim]')
    parser.add_argument('--loss', help='indicate the loss function')
    parser.add_argument('--cnstrnts', help='indicate the constraint scheme')
    parser.add_argument('--lr', default=float(1e-3), type=float, help='indicate the learning rate of the optimizer')
    parser.add_argument('--wdecay', default=float(1e-5), type=float, help='indicate the weight decay of the optimizer')
    parser.add_argument('--lmcoef', default=0.5, type=float, help='indicate the coefficient of the language model loss when fine tuning')
    parser.add_argument('--sampfrac', type=float, help='indicate the sampling fraction for datasets')
    parser.add_argument('--pdrop', default=0.2, type=float, help='indicate the dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler')
    parser.add_argument('--pthrshld', default=0.5, type=float, help='indicate the threshold for predictive probabilitiy')
    parser.add_argument('--topk', default=5, type=int, help='indicate the top k search parameter')
    parser.add_argument('--do_tfidf', default=False, action='store_true', help='whether to use tfidf as text features')
    parser.add_argument('--do_chartfidf', default=False, action='store_true', help='whether to use charater tfidf as text features')
    parser.add_argument('--do_bm25', default=False, action='store_true', help='whether to use bm25 as text features')
    parser.add_argument('--clipmaxn', type=float, help='indicate the max norm of the gradients')
    parser.add_argument('--resume', help='resume training model file')
    parser.add_argument('--refresh', default=False, action='store_true', help='refresh the trained model with newest code')
    parser.add_argument('--sampw', default=False, action='store_true', dest='sample_weights', help='use sample weights')
    parser.add_argument('--corpus', help='corpus data')
    parser.add_argument('--onto', help='ontology data')
    parser.add_argument('--pred', help='prediction file')
    parser.add_argument('--datapath', help='location of dataset')
    parser.add_argument('-i', '--input', help='input dataset')
    parser.add_argument('-w', '--cache', default='.cache', help='the location of cache files')
    parser.add_argument('-u', '--task', default='ddi', help='the task name [default: %default]')
    parser.add_argument('--model', default='bert', help='the model to be validated')
    parser.add_argument('--encoder', choices=['pool', 's2v', 's2s'], help='the encoder to be used after the language model: pool, s2v or s2s')
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model file')
    parser.add_argument('--seed', help='manually set the random seed')
    parser.add_argument('--dfsep', default='\t', help='separate character for pandas dataframe')
    parser.add_argument('--sc', default=';;', help='separate character for multiple-value records')
    parser.add_argument('--configfmt', choices=['json', 'yaml'], default='json', help='config data stored format: json or yaml [default: %default]')
    parser.add_argument('-c', '--cfg', help='config string used to update the settings, format: {\'param_name1\':param_value1[, \'param_name1\':param_value1]}')
    parser.add_argument('-m', '--method', default='classify', help='main method to run')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', dest='verbose', help='display detailed information')
    args = parser.parse_args()

    # Logging setting
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Parse config file
    if (os.path.exists(CONFIG_FILE)):
    	cfgr = io.cfg_reader(CONFIG_FILE)
    else:
        logging.error('Config file `%s` does not exist!' % CONFIG_FILE)
        sys.exit(1)

    # Update config
    cfg_kwargs = {} if args.cfg is None else ast.literal_eval(args.cfg)
    args.cfg = cfg_kwargs
    _update_cfgs(cfg_kwargs)

    # GPU setting
    if (args.gpuq is not None and not args.gpuq.strip().isspace()):
    	args.gpuq = list(range(torch.cuda.device_count())) if (args.gpuq == 'auto' or args.gpuq == 'all') else [int(x) for x in args.gpuq.split(',') if x]
    elif (args.gpunum > 0):
        args.gpuq = list(range(args.gpunum))
    else:
        args.gpuq = []
    if (args.gpuq and args.gpunum > 0):
        if args.verbose: os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[:args.gpunum]) if 'CUDA_VISIBLE_DEVICES' in os.environ and not os.environ['CUDA_VISIBLE_DEVICES'].isspace() else ','.join(map(str, args.gpuq[:args.gpunum]))
        setattr(args, 'devq', list(range(torch.cuda.device_count())))
    else:
        setattr(args, 'devq', None)
    if args.distrb:
        import horovod.torch as hvd
        hvd.init()
        DATA_PATH = os.path.join('/', 'data', 'bionlp')
        torch.cuda.set_device(hvd.local_rank())

    # Process config
    if args.datapath is not None: DATA_PATH = args.datapath
    SC = args.sc

    # Random seed setting
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # Format some arguments
    args.output_layer = list(map(int, args.output_layer.split(',')))
    args.output_layer = args.output_layer[0] if len(args.output_layer) == 1 else args.output_layer
    args.clswfac = list(map(float, args.clswfac.split(','))) if ',' in args.clswfac else float(args.clswfac)

    main()
