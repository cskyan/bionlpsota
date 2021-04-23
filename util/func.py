#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2021 by Caspar. All rights reserved.
# File Name: func.py
# Author: Shankai Yan
# E-mail: dr.skyan@gmail.com
# Created Time: 2021-03-29 17:19:30
###########################################################################
#

import os, sys, copy

import scipy as sp

import torch
from torch import nn

from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper

from bionlp.util import io
from . import config as C
from .dataset import DataParallel


def gen_pytorch_wrapper(mdl_type, mdl_name, **kwargs):
    wrapper_cls = PytorchSeq2SeqWrapper if mdl_type == 'seq2seq' else PytorchSeq2VecWrapper
    mdl_cls = C.PYTORCH_WRAPPER[mdl_name]
    return wrapper_cls(module=mdl_cls(**kwargs))


def gen_mdl(mdl_name, config, pretrained=True, use_gpu=False, distrb=False, dev_id=None):
    if mdl_name == 'none': return None
    wsdir = config.wsdir if hasattr(config, 'wsdir') and os.path.isdir(config.wsdir) else '.'
    if distrb: import horovod.torch as hvd
    if (type(pretrained) is str):
        if (not distrb or distrb and hvd.rank() == 0): print('Using pretrained model from `%s`' % pretrained)
        checkpoint = torch.load(pretrained, map_location='cpu')
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
    elif (pretrained):
        if (not distrb or distrb and hvd.rank() == 0): print('Using pretrained model...')
        mdl_name = mdl_name.split('_')[0]
        common_cfg = config.common_cfg if hasattr(config, 'common_cfg') else {}
        pr = io.param_reader(os.path.join(wsdir, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
        params = pr('LM', config.lm_params)
        model = config.lm_model.from_pretrained(params['pretrained_mdl_path'] if 'pretrained_mdl_path' in params else config.lm_mdl_name)
    else:
        if (not distrb or distrb and hvd.rank() == 0): print('Using untrained model...')
        try:
            common_cfg = config.common_cfg if hasattr(config, 'common_cfg') else {}
            pr = io.param_reader(os.path.join(wsdir, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
            params = pr('LM', config.lm_params)
            for pname in ['pretrained_mdl_path', 'pretrained_vocab_path']:
                if pname in params: del params[pname]
            lm_config = config.lm_config(**params)
            if (mdl_name == 'elmo'):
                pos_params = [lm_config[k] for k in ['options_file','weight_file', 'num_output_representations']]
                kw_params = dict([(k, lm_config[k]) for k in lm_config.keys() if k not in ['options_file','weight_file', 'num_output_representations', 'elmoedim']])
                print('ELMo model parameters: %s %s' % (pos_params, kw_params))
                model = config.lm_model(*pos_params, **kw_params)
            else:
                model = config.lm_model(lm_config)
        except Exception as e:
            print(e)
            print('Cannot find the pretrained model file, using online model instead.')
            model = config.lm_model.from_pretrained(config.lm_mdl_name)
    if (use_gpu): model = model.to('cuda')
    return model


def gen_clf(mdl_name, config, encoder='pool', constraints=[], use_gpu=False, distrb=False, dev_id=None, **kwargs):
    lm_mdl_name = mdl_name.split('_')[0]
    kwargs['lm_mdl_name'] = lm_mdl_name
    common_cfg = config.common_cfg if hasattr(config, 'common_cfg') else {}
    wsdir = config.wsdir if hasattr(config, 'wsdir') and os.path.isdir(config.wsdir) else '.'
    pr = io.param_reader(os.path.join(wsdir, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
    params = pr('LM', config.lm_params) if lm_mdl_name != 'none' else {}
    for pname in ['pretrained_mdl_path', 'pretrained_vocab_path']:
        if pname in params: del params[pname]
    kwargs['config'] = config.lm_config(**params) if lm_mdl_name != 'none' else {}

    lvar = locals()
    for x in constraints:
        cnstrnt_cls, cnstrnt_params = copy.deepcopy(C.CNSTRNTS_MAP[x])
        constraint_params = pr('Constraint', C.CNSTRNT_PARAMS_MAP[x])
        cnstrnt_params.update(dict([((k, p), constraint_params[p]) for k, p in cnstrnt_params.keys() if p in constraint_params]))
        cnstrnt_params.update(dict([((k, p), kwargs[p]) for k, p in cnstrnt_params.keys() if p in kwargs]))
        cnstrnt_params.update(dict([((k, p), lvar[p]) for k, p in cnstrnt_params.keys() if p in lvar]))
        kwargs.setdefault('constraints', []).append((cnstrnt_cls, dict([(k, v) for (k, p), v in cnstrnt_params.items()])))

    clf = config.clf[encoder](**kwargs) if hasattr(config, 'embed_type') and config.embed_type else config.clf(**kwargs)
    if use_gpu: clf = _handle_model(clf, dev_id=dev_id, distrb=distrb)
    return clf


def save_model(model, optimizer, fpath='checkpoint.pth', in_wrapper=False, devq=None, distrb=False, **kwargs):
    print('Saving trained model...')
    use_gpu, multi_gpu = (devq and len(devq) > 0), (devq and len(devq) > 1)
    if not distrb and (in_wrapper or multi_gpu): model = model.module
    model = model.cpu() if use_gpu else model
    checkpoint = {'model': model, 'state_dict': model.state_dict(), 'optimizer':optimizer if not distrb else None, 'optimizer_state_dict':optimizer.state_dict()}
    checkpoint.update(kwargs)
    torch.save(checkpoint, fpath)
    model = _handle_model(model, dev_id=devq, distrb=distrb) if use_gpu else model


def load_model(mdl_path):
    print('Loading previously trained model...')
    checkpoint = torch.load(mdl_path, map_location='cpu')
    model, optimizer = checkpoint['model'], checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint.setdefault('resume', {}), dict([(k, v) for k, v in checkpoint.items() if k not in ['model', 'state_dict', 'optimizer', 'optimizer_state_dict', 'resume']])


def _weights_init(mean=0., std=0.02):
    def _wi(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(mean, std)
        elif classname.find('Linear') != -1 or classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, mean, std)
            nn.init.normal_(m.bias, 0)
            # m.weight.data.normal_(mean, std)
            # m.bias.data.normal_(0)
            # m.bias.data.fill_(0)
    return _wi


def _sprmn_cor(trues, preds):
    return sp.stats.spearmanr(trues, preds)[0]


def _prsn_cor(trues, preds):
    return sp.stats.pearsonr(trues, preds)[0]


def _handle_model(model, dev_id=None, distrb=False):
    if (distrb):
        model.cuda()
    elif (dev_id is not None):
        if (type(dev_id) is list):
            model = model.to('cuda')
            model = DataParallel(model, device_ids=dev_id)
        else:
            torch.cuda.set_device(dev_id)
            model = model.to('cuda')
    return model


def _update_cfgs(cfgs):
    global_vars = globals()
    for glb, glbvs in cfgs.items():
        if glb in global_vars:
            if type(global_vars[glb]) is dict:
                for cfgk in [opts.task, opts.model, opts.method]:
                    if cfgk in global_vars[glb]:
                        if type(global_vars[glb][cfgk]) is dict:
                            global_vars[glb][cfgk].update(glbvs)
                        else:
                            global_vars[glb][cfgk] = glbvs
            else:
                global_vars[glb] = glbvs
