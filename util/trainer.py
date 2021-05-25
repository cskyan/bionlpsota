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


def train(clf, optimizer, dataset, config, scheduler=None, weights=None, lmcoef=0.5, clipmaxn=0.25, epochs=1, earlystop=False, earlystop_delta=0.005, earlystop_patience=5, use_gpu=False, devq=None, distrb=False, resume={}, chckpnt_kwargs={}):
    if distrb: import horovod.torch as hvd
    chckpnt_fname, model_fname = '%s_%s_checkpoint.pth' % (config.dataset, config.model), '%s_%s.pth' % (config.dataset, config.model)
    earlystoper = EarlyStopping(mode='min', min_delta=earlystop_delta, patience=earlystop_patience)
    clf.train()
    clf.unfreeze_lm()
    killer = system.GracefulKiller()
    in_wrapper = type(clf) is DataParallel
    elapsed_batch = resume.setdefault('batch', 0)
    for epoch in range(resume.setdefault('epoch', 0), epochs):
        t0 = time.time()
        total_loss = 0
        for step, batch in enumerate(tqdm(dataset, desc='[%i/%i epoch(s)] Training batches' % (epoch + 1, epochs), disable=distrb and hvd.rank()!=0)):
            try:
                if epoch == resume['epoch'] and step < elapsed_batch: continue
                optimizer.zero_grad()
                # Obtain data
                idx, tkns_tnsr, lb_tnsr = batch[:3]
                extra_inputs = batch[3:]
                if config.task_type == 'nmt':
                    record_idx, extra_inputs = extra_inputs[0], extra_inputs[1:]
                    record_idx = [list(map(int, x.split(config.sc))) for x in record_idx]
                if len(idx) < 2: continue
                # Move to GPUs
                if (use_gpu): tkns_tnsr, lb_tnsr, weights, extra_inputs = [tkns_tnsr[x].to('cuda') for x in [0,1]] if config.task_type in ['entlmnt', 'sentsim'] and clf.mlt_trnsfmr else tkns_tnsr.to('cuda'), lb_tnsr.to('cuda'), (weights if weights is None else weights.to('cuda')), [x.to('cuda') for x in extra_inputs]
                # Calculate loss
                clf_loss, lm_loss, extra_outputs = clf(tkns_tnsr, *extra_inputs, labels=lb_tnsr, weights=weights)
                clf_loss = ((clf_loss.view(tkns_tnsr.size(0), -1) * mask_tnsr.float()).sum(1) / (1e-12 + mask_tnsr.float().sum(1))).mean() if config.task_type == 'nmt' and clf.crf is None else clf_loss.mean()
                train_loss = clf_loss if lm_loss is None else (clf_loss + lmcoef * ((lm_loss.view(tkns_tnsr.size(0), -1) * lm_mask_tnsr.float()).sum(1) / (1e-12 + lm_mask_tnsr.float().sum(1))).mean())
                total_loss += train_loss.item()
                # Backward propagate updates
                train_loss.backward()
                if (clipmaxn is not None):
                    nn.utils.clip_grad_value_(clf.lm_model.encoder.layer.parameters() if config.model.startswith('bert') else clf.parameters(), clipmaxn)
                    nn.utils.clip_grad_value_(clf.parameters(), clipmaxn)
                optimizer.step()
                if scheduler: scheduler.step()
            except Exception as e:
                logging.warning(e)
                train_time = time.time() - t0
                logging.error('Exception raised! training time for %i epoch(s), %i batch(s): %0.3fs' % (epoch + 1, step, train_time))
                save_model(clf, optimizer, chckpnt_fname, in_wrapper=in_wrapper, devq=devq, distrb=config.distrb, resume={'epoch':epoch, 'batch':step}, **chckpnt_kwargs)
                raise e
            # Save model when program is terminated
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
        # Earlystop
        if earlystop:
            do_stop = earlystoper.step(avg_loss)
            if distrb: do_stop = hvd.broadcast(torch.tensor(do_stop).type(torch.ByteTensor), 0).type(torch.BoolTensor)
            if do_stop:
                logging.info('Early stop!')
                break
    # Save model
    try:
        if (not distrb or distrb and hvd.rank() == 0):
            if os.path.exists(chckpnt_fname): os.remove(chckpnt_fname)
            save_model(clf, optimizer, model_fname, devq=devq, distrb=config.distrb)
    except Exception as e:
        logging.warning(e)


def eval(clf, dataset, config, ds_name='', clipmaxn=0.25, use_gpu=False, devq=None, distrb=False, ignored_label=None):
    clf.eval()
    clf.freeze_lm()
    total_loss, indices, preds, probs, all_logits, trues, config.dataset = 0, [], [], [], [], [], config.dataset.strip()
    if config.task_type not in ['entlmnt', 'sentsim', 'mltl-clf']: dataset.dataset.remove_mostfrqlb()
    for step, batch in enumerate(tqdm(dataset, desc="%s batches" % ds_name.title() if ds_name else 'Evaluation')):
        # Obtain data
        idx, tkns_tnsr, lb_tnsr = batch[:3]
        extra_inputs = batch[3:]
        if config.task_type == 'nmt':
            record_idx, extra_inputs = extra_inputs[0], extra_inputs[1:]
            record_idx = [list(map(int, x.split(config.sc))) for x in record_idx]
        extra_inputs = tuple([x.to('cuda') if isinstance(x, torch.Tensor) and use_gpu else x for x in extra_inputs])
        indices.extend(idx if type(idx) is list else (idx.tolist() if type(idx) is torch.Tensor else list(idx)))
        _lb_tnsr = lb_tnsr
        # Caculate output digits
        with torch.no_grad(): outputs = clf(tkns_tnsr, *extra_inputs, labels=None)
        logits, extra_outputs = outputs
        # Caculate loss and probabilitiy
        if config.task_type == 'mltc-clf' or (config.task_type == 'entlmnt' and len(clf.binlbr) > 1):
            loss_func = nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits.view(-1, len(clf.binlbr)), lb_tnsr.view(-1))
            prob, pred = torch.softmax(logits, -1).max(-1)
        elif config.task_type == 'mltl-clf' or (config.task_type == 'entlmnt' and len(clf.binlbr) == 1):
            loss_func = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_func(logits.view(-1, len(clf.binlbr)), lb_tnsr.view(-1, len(clf.binlbr)).float())
            prob = torch.sigmoid(logits).data.view(-1, len(clf.binlbr))
            pred = (prob > (clf.thrshld if config.do_thrshld else config.pthrshld)).int()
        elif config.task_type == 'nmt':
            loss_func = nn.CrossEntropyLoss(reduction='none')
            loss = loss_func(logits.view(-1, len(clf.binlbr)), lb_tnsr.view(-1))
            prob, pred = torch.softmax(logits, -1).max(-1)
        elif config.task_type == 'sentsim':
            loss_func = nn.MSELoss(reduction='none')
            loss = loss_func(logits.view(-1), lb_tnsr.view(-1))
            prob, pred = logits, logits
        total_loss += loss.mean().item()
        # Collect labels and predictions
        trues.append(_lb_tnsr.view(_lb_tnsr.size(0), -1).detach().cpu().numpy() if config.task_type == 'mltl-clf' else _lb_tnsr.view(-1).detach().cpu().numpy())
        preds.append(pred.view(pred.size(0), -1).detach().cpu().numpy() if config.task_type == 'mltl-clf' else pred.view(-1).detach().cpu().numpy())
        probs.append(prob.view(prob.size(0), -1).detach().cpu().numpy() if config.task_type == 'mltl-clf' else prob.view(-1).detach().cpu().numpy())
        all_logits.append(logits.view(_lb_tnsr.size(0), -1, logits.size(-1)).detach().cpu().numpy())
    total_loss = total_loss / (step + 1)
    logging.info('Evaluation loss on %s dataset: %.2f' % (config.dataset, total_loss))

    # Save raw outputs
    all_logits = np.concatenate(all_logits, axis=0)
    if config.task_type == 'nmt':
        if (type(indices[0]) is str and config.sc in indices[0]):
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
    resf_prefix = config.dataset.lower().replace(' ', '_')
    with open('%s_preds_trues.pkl' % resf_prefix, 'wb') as fd:
        pickle.dump((trues, preds, probs, all_logits), fd)
    # Calculate performance
    if any(config.task_type == t for t in ['mltc-clf', 'entlmnt', 'nmt']):
        preds = preds
    elif config.task_type == 'mltl-clf':
        preds = preds
    elif config.task_type == 'sentsim':
        preds = np.squeeze(preds)
    if config.task_type == 'sentsim':
        if (np.isnan(preds).any()):
            logging.info('Predictions contain NaN values! Please try to decrease the learning rate!')
            return
        try:
            metric_names, metrics_funcs = ['Mean Absolute Error', 'Mean Squared Error', 'Mean Squared Log Error', 'Median Absolute Error', 'R2', 'Spearman Correlation', 'Pearson Correlation'], [metrics.mean_absolute_error, metrics.mean_squared_error, metrics.mean_squared_log_error, metrics.median_absolute_error, metrics.r2_score, _sprmn_cor, _prsn_cor]
            perf_df = pd.DataFrame(dict([(k, [f(trues, preds)]) for k, f in zip(metric_names, metrics_funcs)]), index=[config.model])[metric_names]
        except Exception as e:
            logging.warning(e)
            metric_names, metrics_funcs = ['Mean Absolute Error', 'Median Absolute Error', 'R2', 'Spearman Correlation', 'Pearson Correlation'], [metrics.mean_absolute_error, metrics.median_absolute_error, metrics.r2_score, _sprmn_cor, _prsn_cor]
            perf_df = pd.DataFrame(dict([(k, [f(trues, preds)]) for k, f in zip(metric_names, metrics_funcs)]), index=[config.model])[metric_names]
    elif config.task_type in ['mltl-clf', 'nmt']:
        labels = list(clf.binlbr.keys()-[clf.binlb[ignored_label]]) if ignored_label else list(clf.binlbr.keys())
        perf_df = pd.DataFrame(metrics.classification_report(trues, preds, labels=labels, target_names=[clf.binlbr[x] for x in labels], output_dict=True)).T[['precision', 'recall', 'f1-score', 'support']]
    else:
        labels = [x for x in (list(clf.binlbr.keys()-[clf.binlb[ignored_label]]) if ignored_label else list(clf.binlbr.keys())) if x in preds or x in trues] if len(clf.binlbr) > 1 else [0, 1]
        perf_df = pd.DataFrame(metrics.classification_report(trues, preds, labels=labels, target_names=[clf.binlbr[x] for x in labels], output_dict=True)).T[['precision', 'recall', 'f1-score', 'support']]
    logging.info('Results for %s dataset is:\n%s' % (config.dataset.title(), perf_df))
    perf_df.to_excel('perf_%s.xlsx' % resf_prefix)

    try:
        dataset.dataset.fill_labels(preds, saved_path='pred_%s.csv' % resf_prefix, index=indices)
    except Exception as e:
        raise e
        logging.warning(e)
