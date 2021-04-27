#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2021 by Caspar. All rights reserved.
# File Name: trainer.py
# Author: Shankai Yan
# E-mail: dr.skyan@gmail.com
# Created Time: 2021-03-29 17:19:30
###########################################################################
#

import os, sys, time, copy, pickle, itertools, logging
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch import nn

from sklearn import metrics

from bionlp.util import system

from modules.common import EarlyStopping
from .dataset import DataParallel
from .processor import nlp, _batch2ids_w2v
from .func import _sprmn_cor, _prsn_cor, save_model


def train(clf, optimizer, dataset, config, special_tkns, scheduler=None, pad_val=0, weights=None, lmcoef=0.5, clipmaxn=0.25, epochs=1, earlystop=False, earlystop_delta=0.005, earlystop_patience=5, task_type='mltc-clf', task_name='classification', mdl_name='sota', use_gpu=False, devq=None, distrb=False, resume={}, chckpnt_kwargs={}):
    if distrb: import horovod.torch as hvd
    chckpnt_fname, model_fname = '%s_%s_checkpoint.pth' % (task_name, mdl_name), '%s_%s.pth' % (task_name, mdl_name)
    clf_tknids = special_tkns['clf_tknids']
    earlystoper = EarlyStopping(mode='min', min_delta=earlystop_delta, patience=earlystop_patience)
    clf.train()
    clf.unfreeze_lm()
    killer = system.GracefulKiller()
    in_wrapper = type(clf) is DataParallel
    elapsed_batch = resume.setdefault('batch', 0)
    for epoch in range(resume.setdefault('epoch', 0), epochs):
        t0 = time.time()
        total_loss = 0
        # if task_type not in ['entlmnt', 'sentsim']: dataset.dataset.rebalance()
        for step, batch in enumerate(tqdm(dataset, desc='[%i/%i epoch(s)] Training batches' % (epoch + 1, epochs), disable=distrb and hvd.rank()!=0)):
            try:
                if epoch == resume['epoch'] and step < elapsed_batch: continue
                optimizer.zero_grad()
                idx, tkns_tnsr, lb_tnsr = batch[:3]
                extra_inputs = batch[3:]
                if task_type == 'nmt':
                    record_idx, extra_inputs = extra_inputs[0], extra_inputs[1:]
                    record_idx = [list(map(int, x.split(config.sc))) for x in record_idx]
                # extra_inputs = tuple([x.to('cuda') if use_gpu else x for x in extra_inputs])
                if len(idx) < 2: continue
                if hasattr(config, 'embed_type') and config.embed_type:
                    if task_type in ['entlmnt', 'sentsim']:
                        tkns_tnsr = [[[w.text for w in nlp(sents)] + special_tkns['delim_tknids'] for sents in tkns_tnsr[x]] for x in [0,1]]
                        tkns_tnsr = [[s[:min(len(s), config.maxlen)] for s in tkns_tnsr[x]] for x in [0,1]]
                        w2v_tnsr = [_batch2ids_w2v([s + [''] * (config.maxlen-len(s)) for s in tkns_tnsr[x]], clf.w2v_model) for x in [0,1]] if hasattr(clf, 'w2v_model') and clf.w2v_model else None
                        sentvec_tnsr = [_batch2ids_sentvec(tkns_tnsr[x], clf.w2v_model) for x in [0,1]] if hasattr(clf, 'sentvec_model') and clf.sentvec_model else None
                        pool_idx = [torch.LongTensor([len(s) - 1 for s in tkns_tnsr[x]]) for x in [0,1]]
                        if mdl_name.startswith('elmo'):
                            tkns_tnsr = [tkns_tnsr[x] + [[''] * config.maxlen] for x in [0,1]]
                            tkns_tnsr = [batch_to_ids(tkns_tnsr[x])[:-1] for x in [0,1]]
                            pad_val = 0
                        else:
                            # tkns_tnsr = [[s + [''] * (config.maxlen-len(s)) for s in tkns_tnsr[x]] for x in [0,1]]
                            tkns_tnsr = [torch.tensor([[1]*len(s) + [pad_val[0]] * (config.maxlen-len(s)) for s in tkns_tnsr[x]]) for x in [0,1]]
                        if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx, weights, extra_inputs = [tkns_tnsr[x].to('cuda') for x in [0,1]] , lb_tnsr.to('cuda'), [pool_idx[x].to('cuda') for x in [0,1]], (weights if weights is None else weights.to('cuda')), [x.to('cuda') for x in extra_inputs]
                    elif task_type == 'nmt':
                        tkns_tnsr, lb_tnsr = [s.split(config.sc) for s in tkns_tnsr if (type(s) is str and s != '') and len(s) > 0], [list(map(int, s.split(config.sc))) for s in lb_tnsr if (type(s) is str and s != '') and len(s) > 0]
                        # tkns_tnsr, lb_tnsr = zip(*[(sx.split(config.sc), list(map(int, sy.split(config.sc)))) for sx, sy in zip(tkns_tnsr, lb_tnsr) if ((type(sx) is str and sx != '') or len(sx) > 0) and ((type(sy) is str and sy != '') or len(sy) > 0)])
                        if (len(tkns_tnsr) == 0 or len(lb_tnsr) == 0): continue
                        lb_tnsr = torch.LongTensor([s[:min(len(s), config.maxlen)] + [pad_val[1]] * (config.maxlen-len(s)) for s in lb_tnsr])
                        tkns_tnsr = [s[:min(len(s), config.maxlen)] for s in tkns_tnsr]
                        w2v_tnsr = _batch2ids_w2v([s + [''] * (config.maxlen-len(s)) for s in tkns_tnsr], clf.w2v_model) if hasattr(clf, 'w2v_model') and clf.w2v_model else None
                        sentvec_tnsr = None
                        pool_idx = torch.LongTensor([len(s) - 1 for s in tkns_tnsr])
                        if mdl_name.startswith('elmo'):
                            tkns_tnsr = tkns_tnsr + [[''] * config.maxlen]
                            tkns_tnsr = batch_to_ids(tkns_tnsr)[:-1]
                            pad_val = (0, pad_val[1])
                        else:
                            # tkns_tnsr = [s + [''] * (config.maxlen-len(s)) for s in tkns_tnsr]
                            tkns_tnsr = torch.tensor([[1]*len(s) + [pad_val] * (config.maxlen-len(s)) for s in tkns_tnsr])
                        if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx, weights, extra_inputs = tkns_tnsr.to('cuda') , lb_tnsr.to('cuda'), pool_idx.to('cuda'), (weights if weights is None else weights.to('cuda')), [x.to('cuda') for x in extra_inputs]
                    else:
                        tkns_tnsr = [[w.text for w in nlp(text)] for text in tkns_tnsr]
                        tkns_tnsr = [s[:min(len(s), config.maxlen)] for s in tkns_tnsr]
                        w2v_tnsr = _batch2ids_w2v([s + [''] * (config.maxlen-len(s)) for s in tkns_tnsr], clf.w2v_model) if hasattr(clf, 'w2v_model') and clf.w2v_model else None
                        sentvec_tnsr = _batch2ids_sentvec(tkns_tnsr, clf.sentvec_model) if hasattr(clf, 'sentvec_model') and clf.sentvec_model else None
                        pool_idx = torch.LongTensor([len(s) - 1 for s in tkns_tnsr])
                        if mdl_name.startswith('elmo'):
                            tkns_tnsr = tkns_tnsr + [[''] * config.maxlen]
                            tkns_tnsr = batch_to_ids(tkns_tnsr)[:-1]
                            pad_val = 0
                        else:
                            # tkns_tnsr = [s + [''] * (config.maxlen-len(s)) for s in tkns_tnsr]
                            tkns_tnsr = torch.tensor([[1]*len(s) + [pad_val] * (config.maxlen-len(s)) for s in tkns_tnsr])
                        if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx, weights, extra_inputs = tkns_tnsr.to('cuda') , lb_tnsr.to('cuda'), pool_idx.to('cuda'), (weights if weights is None else weights.to('cuda')), [x.to('cuda') for x in extra_inputs]
                    if mdl_name.endswith('sentvec'): extra_inputs += (sentvec_tnsr,)
                    mask_tnsr = [(~tkns_tnsr[x].eq(pad_val * torch.ones_like(tkns_tnsr[x]))).long() for x in [0,1]] if task_type in ['entlmnt', 'sentsim'] else (~tkns_tnsr.eq(pad_val[0] if task_type=='nmt' else pad_val * torch.ones_like(tkns_tnsr))).long()
                    clf_loss, lm_loss = clf(tkns_tnsr, pool_idx, *extra_inputs, w2v_ids=w2v_tnsr, labels=lb_tnsr, weights=weights)
                else:
                    valid_idx = [x for x in range(tkns_tnsr.size(0)) if x not in np.transpose(np.argwhere(tkns_tnsr == -1))[:,0]]
                    if (len(valid_idx) == 0): continue
                    idx, tkns_tnsr, lb_tnsr, record_idx = [idx[x] for x in range(len(idx)) if x in valid_idx], tkns_tnsr[valid_idx], lb_tnsr[valid_idx], [record_idx[x] for x in range(len(record_idx)) if x in valid_idx] if task_type == 'nmt' else None
                    tkns_tnsr = [tkns_tnsr[:,x,:] for x in [0,1]] if task_type in ['entlmnt', 'sentsim'] and clf.mlt_trnsfmr else tkns_tnsr
                    mask_tnsr = [(~tkns_tnsr[x].eq(pad_val * torch.ones_like(tkns_tnsr[x]))).long() for x in [0,1]] if task_type in ['entlmnt', 'sentsim'] and clf.mlt_trnsfmr else (~tkns_tnsr.eq(pad_val[0] if task_type=='nmt' else pad_val * torch.ones_like(tkns_tnsr))).long()
                    lm_mask_tnsr = mask_tnsr if mdl_name in ['bert', 'trsfmxl'] else ([mask_tnsr[x][:, 1:].contiguous().view(tkns_tnsr[x].size(0), -1) for x in [0,1]] if task_type in ['entlmnt', 'sentsim'] and clf.mlt_trnsfmr else mask_tnsr[:, 1:].contiguous())
                    if (use_gpu): tkns_tnsr, lb_tnsr, lm_mask_tnsr, mask_tnsr, weights, extra_inputs = [tkns_tnsr[x].to('cuda') for x in [0,1]] if task_type in ['entlmnt', 'sentsim'] and clf.mlt_trnsfmr else tkns_tnsr.to('cuda'), lb_tnsr.to('cuda'), (lm_mask_tnsr if lm_mask_tnsr is None else ([lm_mask_tnsr[x].to('cuda') for x in [0,1]] if task_type in ['entlmnt', 'sentsim'] and clf.mlt_trnsfmr else lm_mask_tnsr.to('cuda'))), [mask_tnsr[x].to('cuda') for x in [0,1]] if task_type in ['entlmnt', 'sentsim'] and clf.mlt_trnsfmr else mask_tnsr.to('cuda'), (weights if weights is None else weights.to('cuda')), [x.to('cuda') for x in extra_inputs]
                    _pool_idx = pool_idx = [tkns_tnsr[x].eq(clf_tknids[0] * torch.ones_like(tkns_tnsr[x])).int().argmax(-1) for x in[0,1]] if task_type in ['entlmnt', 'sentsim'] and clf.mlt_trnsfmr else tkns_tnsr.eq(clf_tknids[0] * torch.ones_like(tkns_tnsr)).int().argmax(-1)
                    pool_idx = mask_tnsr if task_type == 'nmt' or mdl_name.startswith('bert') else pool_idx
                    clf_loss, lm_loss = clf(tkns_tnsr, pool_idx, *extra_inputs, labels=lb_tnsr, weights=weights)
                clf_loss = ((clf_loss.view(tkns_tnsr.size(0), -1) * mask_tnsr.float()).sum(1) / (1e-12 + mask_tnsr.float().sum(1))).mean() if task_type == 'nmt' and clf.crf is None else clf_loss.mean()
                train_loss = clf_loss if lm_loss is None else (clf_loss + lmcoef * ((lm_loss.view(tkns_tnsr.size(0), -1) * lm_mask_tnsr.float()).sum(1) / (1e-12 + lm_mask_tnsr.float().sum(1))).mean())
                total_loss += train_loss.item()
                train_loss.backward()
                if (clipmaxn is not None):
                    nn.utils.clip_grad_value_(clf.lm_model.encoder.layer.parameters() if mdl_name.startswith('bert') else clf.parameters(), clipmaxn)
                    nn.utils.clip_grad_value_(clf.parameters(), clipmaxn)
                optimizer.step()
                if scheduler: scheduler.step()
            except Exception as e:
                logging.warning(e)
                train_time = time.time() - t0
                logging.warning('Exception raised! training time for %i epoch(s), %i batch(s): %0.3fs' % (epoch + 1, step, train_time))
                save_model(clf, optimizer, chckpnt_fname, in_wrapper=in_wrapper, devq=devq, distrb=config.distrb, resume={'epoch':epoch, 'batch':step}, **chckpnt_kwargs)
                raise e
            if (killer.kill_now):
                train_time = time.time() - t0
                logging.info('Interrupted! training time for %i epoch(s), %i batch(s): %0.3fs' % (epoch + 1, step, train_time))
                if (not distrb or distrb and hvd.rank() == 0):
                    save_model(clf, optimizer, chckpnt_fname, in_wrapper=in_wrapper, devq=devq, distrb=config.distrb, resume={'epoch':epoch, 'batch':step}, **chckpnt_kwargs)
                sys.exit(0)
        avg_loss = total_loss / (step + 1)
        logging.info('Train loss in %i epoch(s): %f' % (epoch + 1, avg_loss))
        if epoch % 5 == 0:
            try:
                if (not distrb or distrb and hvd.rank() == 0):
                    save_model(clf, optimizer, chckpnt_fname, in_wrapper=in_wrapper, devq=devq, distrb=config.distrb, resume={'epoch':epoch, 'batch':step}, **chckpnt_kwargs)
            except Exception as e:
                logging.warning(e)
        if earlystop:
            do_stop = earlystoper.step(avg_loss)
            if distrb: do_stop = hvd.broadcast(torch.tensor(do_stop).type(torch.ByteTensor), 0).type(torch.BoolTensor)
            if do_stop:
                logging.info('Early stop!')
                break
    try:
        if (not distrb or distrb and hvd.rank() == 0):
            if os.path.exists(chckpnt_fname): os.remove(chckpnt_fname)
            save_model(clf, optimizer, model_fname, devq=devq, distrb=config.distrb)
    except Exception as e:
        logging.warning(e)


def eval(clf, dataset, config, binlbr, special_tkns, pad_val=0, task_type='mltc-clf', task_name='classification', ds_name='', mdl_name='sota', clipmaxn=0.25, use_gpu=False, devq=None, distrb=False, ignored_label=None):
    clf_tknids = special_tkns['clf_tknids']
    clf.eval()
    clf.freeze_lm()
    total_loss, indices, preds, probs, all_logits, trues, ds_name = 0, [], [], [], [], [], ds_name.strip()
    if task_type not in ['entlmnt', 'sentsim', 'mltl-clf']: dataset.dataset.remove_mostfrqlb()
    for step, batch in enumerate(tqdm(dataset, desc="%s batches" % ds_name.title() if ds_name else 'Evaluation')):
        idx, tkns_tnsr, lb_tnsr = batch[:3]
        extra_inputs = batch[3:]
        if task_type == 'nmt':
            record_idx, extra_inputs = extra_inputs[0], extra_inputs[1:]
            record_idx = [list(map(int, x.split(config.sc))) for x in record_idx]
        extra_inputs = tuple([x.to('cuda') if isinstance(x, torch.Tensor) and use_gpu else x for x in extra_inputs])
        # print(('orig lbtnsr:', lb_tnsr))
        indices.extend(idx if type(idx) is list else (idx.tolist() if type(idx) is torch.Tensor else list(idx)))
        _lb_tnsr = lb_tnsr
        if hasattr(config, 'embed_type') and config.embed_type:
            if task_type in ['entlmnt', 'sentsim']:
                tkns_tnsr = [[[w.text for w in nlp(sents)] + special_tkns['delim_tknids'] for sents in tkns_tnsr[x]] for x in [0,1]]
                tkns_tnsr = [[s[:min(len(s), config.maxlen)] for s in tkns_tnsr[x]] for x in [0,1]]
                w2v_tnsr = [_batch2ids_w2v([s + [''] * (config.maxlen-len(s)) for s in tkns_tnsr[x]], clf.w2v_model) for x in [0,1]] if hasattr(clf, 'w2v_model') and clf.w2v_model else None
                sentvec_tnsr = [_batch2ids_sentvec(tkns_tnsr[x], clf.w2v_model) for x in [0,1]] if hasattr(clf, 'sentvec_model') and clf.sentvec_model else None
                pool_idx = [torch.LongTensor([len(s) - 1 for s in tkns_tnsr[x]]) for x in [0,1]]
                if mdl_name.startswith('elmo'):
                    tkns_tnsr = [tkns_tnsr[x] + [[''] * config.maxlen] for x in [0,1]]
                    tkns_tnsr = [batch_to_ids(tkns_tnsr[x])[:-1] for x in [0,1]]
                    pad_val = 0
                else:
                    # tkns_tnsr = [[s + [''] * (config.maxlen-len(s)) for s in tkns_tnsr[x]] for x in [0,1]]
                    tkns_tnsr = [torch.tensor([[1]*len(s) + [pad_val[0]] * (config.maxlen-len(s)) for s in tkns_tnsr[x]]) for x in [0,1]]
                if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx, w2v_tnsr, sentvec_tnsr, extra_inputs = [tkns_tnsr[x].to('cuda') for x in [0,1]] , lb_tnsr.to('cuda'), [pool_idx[x].to('cuda') for x in [0,1]], w2v_tnsr if w2v_tnsr is None else [w2v_tnsr[x].to('cuda') for x in [0,1]], sentvec_tnsr if sentvec_tnsr is None else [sentvec_tnsr[x].to('cuda') for x in [0,1]], [x.to('cuda') for x in extra_inputs]
            elif task_type == 'nmt':
                tkns_tnsr, lb_tnsr = [s.split(config.sc) for s in tkns_tnsr if (type(s) is str and s != '') and len(s) > 0], [list(map(int, s.split(config.sc))) for s in lb_tnsr if (type(s) is str and s != '') and len(s) > 0]
                # tkns_tnsr, lb_tnsr = zip(*[(sx.split(config.sc), list(map(int, sy.split(config.sc)))) for sx, sy in zip(tkns_tnsr, lb_tnsr) if ((type(sx) is str and sx != '') or len(sx) > 0) and ((type(sy) is str and sy != '') or len(sy) > 0)])
                if (len(tkns_tnsr) == 0 or len(lb_tnsr) == 0): continue
                tkns_tnsr = [s[:min(len(s), config.maxlen)] + [''] * (config.maxlen-len(s)) for s in tkns_tnsr]
                _lb_tnsr = lb_tnsr = torch.LongTensor([s[:min(len(s), config.maxlen)] + [pad_val[1]] * (config.maxlen-len(s)) for s in lb_tnsr])
                tkns_tnsr = [s[:min(len(s), config.maxlen)] for s in tkns_tnsr]
                w2v_tnsr = _batch2ids_w2v([s + [''] * (config.maxlen-len(s)) for s in tkns_tnsr], clf.w2v_model) if hasattr(clf, 'w2v_model') and clf.w2v_model else None
                sentvec_tnsr = None
                _pool_idx = pool_idx = torch.LongTensor([len(s) - 1 for s in tkns_tnsr])
                if mdl_name.startswith('elmo'):
                    tkns_tnsr = tkns_tnsr + [[''] * config.maxlen]
                    tkns_tnsr = batch_to_ids(tkns_tnsr)[:-1]
                    pad_val = (0, pad_val[1])
                else:
                    # tkns_tnsr = [s + [''] * (config.maxlen-len(s)) for s in tkns_tnsr]
                    tkns_tnsr = torch.tensor([[1]*len(s) + [pad_val] * (config.maxlen-len(s)) for s in tkns_tnsr])
                if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx, w2v_tnsr, sentvec_tnsr, extra_inputs = tkns_tnsr.to('cuda'), lb_tnsr.to('cuda'), pool_idx.to('cuda'), (w2v_tnsr if w2v_tnsr is None else w2v_tnsr.to('cuda')), (sentvec_tnsr if sentvec_tnsr is None else sentvec_tnsr.to('cuda')), [x.to('cuda') for x in extra_inputs]
            else:
                tkns_tnsr = [[w.text for w in nlp(text)] for text in tkns_tnsr]
                tkns_tnsr = [s[:min(len(s), config.maxlen)] for s in tkns_tnsr]
                w2v_tnsr = _batch2ids_w2v([s + [''] * (config.maxlen-len(s)) for s in tkns_tnsr], clf.w2v_model) if hasattr(clf, 'w2v_model') and clf.w2v_model else None
                sentvec_tnsr = _batch2ids_sentvec(tkns_tnsr, clf.sentvec_model) if hasattr(clf, 'sentvec_model') and clf.sentvec_model else None
                _pool_idx = pool_idx = torch.LongTensor([len(s) - 1 for s in tkns_tnsr])
                if mdl_name.startswith('elmo'):
                    tkns_tnsr = tkns_tnsr + [[''] * config.maxlen]
                    tkns_tnsr = batch_to_ids(tkns_tnsr)[:-1]
                    pad_val = 0
                else:
                    # tkns_tnsr = [s + [''] * (config.maxlen-len(s)) for s in tkns_tnsr]
                    tkns_tnsr = torch.tensor([[1]*len(s) + [pad_val] * (config.maxlen-len(s)) for s in tkns_tnsr])
                if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx, w2v_tnsr, sentvec_tnsr, extra_inputs = tkns_tnsr.to('cuda') , lb_tnsr.to('cuda'), pool_idx.to('cuda'), (w2v_tnsr if w2v_tnsr is None else w2v_tnsr.to('cuda')), (sentvec_tnsr if sentvec_tnsr is None else sentvec_tnsr.to('cuda')), [x.to('cuda') for x in extra_inputs]
            mask_tnsr = [(~tkns_tnsr[x].eq(pad_val * torch.ones_like(tkns_tnsr[x]))).long() for x in [0,1]] if task_type in ['entlmnt', 'sentsim'] else (~tkns_tnsr.eq(pad_val[0] if task_type=='nmt' else pad_val * torch.ones_like(tkns_tnsr))).long()
            with torch.no_grad():
                outputs = clf(tkns_tnsr, pool_idx, *extra_inputs, w2v_ids=w2v_tnsr, labels=None, **dict(sentvec_tnsr=sentvec_tnsr) if mdl_name.endswith('sentvec') else {})
        else:
            valid_idx = [x for x in range(tkns_tnsr.size(0)) if x not in np.transpose(np.argwhere(tkns_tnsr == -1))[:,0]]
            if (len(valid_idx) == 0): continue
            _, _, _lb_tnsr, _ = idx, tkns_tnsr, lb_tnsr, record_idx = [idx[x] for x in range(len(idx)) if x in valid_idx], tkns_tnsr[valid_idx], lb_tnsr[valid_idx], ([record_idx[x] for x in range(len(record_idx)) if x in valid_idx] if task_type == 'nmt' else None)

            tkns_tnsr = [tkns_tnsr[:,x,:] for x in [0,1]] if task_type in ['entlmnt', 'sentsim'] and clf.mlt_trnsfmr else tkns_tnsr
            mask_tnsr = [(~tkns_tnsr[x].eq(pad_val * torch.ones_like(tkns_tnsr[x]))).long() for x in [0,1]] if task_type in ['entlmnt', 'sentsim'] and clf.mlt_trnsfmr else (~tkns_tnsr.eq(pad_val[0] if task_type=='nmt' else pad_val * torch.ones_like(tkns_tnsr))).long()
            # lm_mask_tnsr = mask_tnsr if mdl_name in ['bert', 'trsfmxl'] else ([mask_tnsr[x][:, :, 1:].contiguous().view(tkns_tnsr[x].size(0), -1) for x in [0,1]] if task_type in ['entlmnt', 'sentsim'] and clf.mlt_trnsfmr else mask_tnsr[:, 1:].contiguous())

            # mask_tnsr = (~tkns_tnsr.eq(pad_val[0] if task_type=='nmt' else pad_val * torch.ones_like(tkns_tnsr))).long()
            # _pool_idx = pool_idx = tkns_tnsr.eq(clf_tknids[0] * torch.ones_like(tkns_tnsr)).int().argmax(-1)
            _pool_idx = pool_idx = [tkns_tnsr[x].eq(clf_tknids[0] * torch.ones_like(tkns_tnsr[x])).int().argmax(-1) for x in[0,1]] if task_type in ['entlmnt', 'sentsim'] and clf.mlt_trnsfmr else tkns_tnsr.eq(clf_tknids[0] * torch.ones_like(tkns_tnsr)).int().argmax(-1)
            if (use_gpu): tkns_tnsr, lb_tnsr, pool_idx, mask_tnsr, extra_inputs = [tkns_tnsr[x].to('cuda') for x in [0,1]] if task_type in ['entlmnt', 'sentsim'] and clf.mlt_trnsfmr else tkns_tnsr.to('cuda'), lb_tnsr.to('cuda'), [pool_idx[x].to('cuda') for x in [0,1]] if task_type in ['entlmnt', 'sentsim'] and clf.mlt_trnsfmr else pool_idx.to('cuda'), [mask_tnsr[x].to('cuda') for x in [0,1]] if task_type in ['entlmnt', 'sentsim'] and clf.mlt_trnsfmr else mask_tnsr.to('cuda'), [x.to('cuda') for x in extra_inputs]
            # print(('tkns, lb, pool, mask:', tkns_tnsr, lb_tnsr, pool_idx, mask_tnsr))
            pool_idx = mask_tnsr if task_type == 'nmt' or mdl_name.startswith('bert') else pool_idx

            with torch.no_grad():
                outputs = clf(tkns_tnsr, pool_idx, *extra_inputs, labels=None)
        logits, extra_outputs = (outputs[0], outputs[1:]) if type(outputs) is tuple else (outputs, None)
        if task_type == 'mltc-clf' or (task_type == 'entlmnt' and len(binlbr) > 1):
            loss_func = nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits.view(-1, len(binlbr)), lb_tnsr.view(-1))
            prob, pred = torch.softmax(logits, -1).max(-1)
        elif task_type == 'mltl-clf' or (task_type == 'entlmnt' and len(binlbr) == 1):
            loss_func = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_func(logits.view(-1, len(binlbr)), lb_tnsr.view(-1, len(binlbr)).float())
            prob = torch.sigmoid(logits).data.view(-1, len(binlbr))
            pred = (prob > (clf.thrshld if config.do_thrshld else config.pthrshld)).int()
            # print(('logits, prob:', logits, prob))
        elif task_type == 'nmt':
            loss_func = nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits.view(-1, len(binlbr)), lb_tnsr.view(-1))
            prob, pred = torch.softmax(logits, -1).max(-1)
        elif task_type == 'sentsim':
            loss_func = nn.MSELoss(reduction='none')
            # print((logits.mean(), lb_tnsr.mean()))
            loss = loss_func(logits.view(-1), lb_tnsr.view(-1))
            prob, pred = logits, logits
        total_loss += loss.mean().item()
        # if task_type == 'nmt':
        #     last_tkns = torch.arange(_lb_tnsr.size(0)) * _lb_tnsr.size(1) + _pool_idx
        #     flat_tures, flat_preds, flat_probs = _lb_tnsr.view(-1).tolist(), pred.view(-1).detach().cpu().tolist(), prob.view(-1).detach().cpu().tolist()
        #     flat_tures_set, flat_preds_set, flat_probs_set = set(flat_tures), set(flat_preds), set(flat_probs)
        #     trues.append([[max(flat_tures_set, key=flat_tures[a:b][c[idx]:c[idx+1]].count) for idx in range(len(c)-1)] for a, b, c in zip(range(_lb_tnsr.size(0)), last_tkns, record_idx)])
        #     preds.append([[max(flat_preds_set, key=flat_preds[a:b][c[idx]:c[idx+1]].count) for idx in range(len(c)-1)] for a, b, c in zip(range(_lb_tnsr.size(0)), last_tkns, record_idx)])
        #     probs.append([[max(flat_probs_set, key=flat_probs[a:b][c[idx]:c[idx+1]].count) for idx in range(len(c)-1)] for a, b, c in zip(range(_lb_tnsr.size(0)), last_tkns, record_idx)])
        #     # preds.append([flat_preds[a:b] for a, b in zip(range(_lb_tnsr.size(0)), last_tkns)])
        #     # probs.append([flat_probs[a:b] for a, b in zip(range(_lb_tnsr.size(0)), last_tkns)])
        # else:
        trues.append(_lb_tnsr.view(_lb_tnsr.size(0), -1).detach().cpu().numpy() if task_type == 'mltl-clf' else _lb_tnsr.view(-1).detach().cpu().numpy())
        preds.append(pred.view(pred.size(0), -1).detach().cpu().numpy() if task_type == 'mltl-clf' else pred.view(-1).detach().cpu().numpy())
        probs.append(prob.view(prob.size(0), -1).detach().cpu().numpy() if task_type == 'mltl-clf' else prob.view(-1).detach().cpu().numpy())

        all_logits.append(logits.view(_lb_tnsr.size(0), -1, logits.size(-1)).detach().cpu().numpy())
    total_loss = total_loss / (step + 1)
    logging.info('Evaluation loss on %s dataset: %.2f' % (ds_name, total_loss))

    all_logits = np.concatenate(all_logits, axis=0)
    if task_type == 'nmt':
        if (type(indices[0]) is str and config.sc in indices[0]):
            # with open('test.pkl', 'wb') as fd:
            #     pickle.dump((indices, trues, preds, probs), fd)
            # indices = list(itertools.chain.from_iterable([list(idx.split(config.sc)) for idx in indices if idx]))
            indices = [list(idx.split(config.sc)) for idx in indices if idx]
            indices, preds, probs, trues = zip(*[(idx, pred, prob, true_lb) for bz_idx, bz_pred, bz_prob, bz_true in zip(indices, preds, probs, trues) for idx, pred, prob, true_lb in zip(*[bz_idx[:min(len(bz_idx), len(bz_pred))], bz_pred[:min(len(bz_idx), len(bz_pred))], bz_prob[:min(len(bz_idx), len(bz_prob))], bz_true[:min(len(bz_idx), len(bz_true))]])])
            indices = map(int, indices) if indices[0].isdigit() else indices
        else:
            preds = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(preds))))
            probs = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(probs))))
            trues = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(trues))))
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
            logging.info('Predictions contain NaN values! Please try to decrease the learning rate!')
            return
        try:
            metric_names, metrics_funcs = ['Mean Absolute Error', 'Mean Squared Error', 'Mean Squared Log Error', 'Median Absolute Error', 'R2', 'Spearman Correlation', 'Pearson Correlation'], [metrics.mean_absolute_error, metrics.mean_squared_error, metrics.mean_squared_log_error, metrics.median_absolute_error, metrics.r2_score, _sprmn_cor, _prsn_cor]
            perf_df = pd.DataFrame(dict([(k, [f(trues, preds)]) for k, f in zip(metric_names, metrics_funcs)]), index=[mdl_name])[metric_names]
        except Exception as e:
            logging.warning(e)
            metric_names, metrics_funcs = ['Mean Absolute Error', 'Median Absolute Error', 'R2', 'Spearman Correlation', 'Pearson Correlation'], [metrics.mean_absolute_error, metrics.median_absolute_error, metrics.r2_score, _sprmn_cor, _prsn_cor]
            perf_df = pd.DataFrame(dict([(k, [f(trues, preds)]) for k, f in zip(metric_names, metrics_funcs)]), index=[mdl_name])[metric_names]
    elif task_type in ['mltl-clf', 'nmt']:
        labels = list(binlbr.keys()-[clf.binlb[ignored_label]]) if ignored_label else list(binlbr.keys())
        perf_df = pd.DataFrame(metrics.classification_report(trues, preds, labels=labels, target_names=[binlbr[x] for x in labels], output_dict=True)).T[['precision', 'recall', 'f1-score', 'support']]
    else:
        labels = [x for x in (list(binlbr.keys()-[clf.binlb[ignored_label]]) if ignored_label else list(binlbr.keys())) if x in preds or x in trues] if len(binlbr) > 1 else [0, 1]
        perf_df = pd.DataFrame(metrics.classification_report(trues, preds, labels=labels, target_names=[binlbr[x] for x in labels], output_dict=True)).T[['precision', 'recall', 'f1-score', 'support']]
    logging.info('Results for %s dataset is:\n%s' % (ds_name.title(), perf_df))
    perf_df.to_excel('perf_%s.xlsx' % resf_prefix)

    try:
        dataset.dataset.fill_labels(preds, saved_path='pred_%s.csv' % resf_prefix, index=indices)
    except Exception as e:
        raise e
        logging.warning(e)
