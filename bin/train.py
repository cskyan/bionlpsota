#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: train.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2019-03-24 02:02:31
###########################################################################
#

import os, sys, time, copy, logging
from optparse import OptionParser
from tqdm import tqdm

import numpy as np

import torch
from torch import nn

from pytorch_pretrained_bert import OpenAIGPTConfig, OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from pytorch_pretrained_bert import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

from bionlp.util import io, system


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
MODEL_NAME_MAP = {'gpt2':'gpt2', 'gpt':'openai-gpt'}
MODEL_MAP = {'gpt2':GPT2LMHeadModel, 'gpt':OpenAIGPTLMHeadModel}
CONFIG_MAP = {'gpt2':GPT2Config, 'gpt':OpenAIGPTConfig}
TKNZR_MAP = {'gpt2':GPT2Tokenizer, 'gpt':OpenAIGPTTokenizer}
SC=';;'

opts, args = {}, []
cfgr = None


def load_data(data_path, model_name, tokenizer=None, level='word'):
    if (data_path and os.path.isfile(data_path)):
        filesize, cumfs = os.path.getsize(data_path), 0.
        pbar = tqdm(total=filesize, desc='Corpus')
        with open(opts.input) as f:
           line = f.readline()
           while line:
               if (model_name == 'gpt'):
                   if tokenizer:
                       tokens = tokenizer.tokenize(line.strip())
                       records = []
                       while (len(tokens) > 512):
                           records.extend(tokenizer.convert_tokens_to_ids(tokens[:512]))
                           tokens = tokens[512:]
                       records.extend(tokenizer.convert_tokens_to_ids(tokens))
                   else:
                       records = line
               elif (model_name == 'gpt2'):
                   records = tokenizer.encode(line.strip()) if tokenizer else line
               if (level == 'word'):
                   words = records if tokenizer else records.strip().split()
                   for w in words:
                       yield w
               elif (level == 'sent'):
                   yield records
               linesize = sys.getsizeof(line)
               cumfs += linesize
               if (cumfs / filesize > 0.001):
                   pbar.update(cumfs)
                   cumfs = 0.
               line = f.readline()
        pbar.close()
    else:
        yield []


def _padtrim(batch_tkns, max_len, pad_val=0):
    batch_tkns = batch_tkns if type(batch_tkns) is list else [batch_tkns]
    num_seq = len(batch_tkns)
    padtrim_X = np.ones((num_seq, max_len), dtype=np.int64) * pad_val
    padtrim_M = np.zeros((num_seq, max_len), dtype=np.float32)
    for i, x in enumerate(batch_tkns):
        seqlen = min(max_len, len(x))
        padtrim_X[i, :seqlen] = x[:seqlen]
        padtrim_M[i, :seqlen] = 1
    return padtrim_X, padtrim_M


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


def _handle_model(model, dev_id=None, distrb=False):
    if (distrb):
        if (type(dev_id) is list):
            model.cuda()
            model = nn.parallel.DistributedDataParallel(model, device_ids=dev_id)
        else:
            torch.cuda.set_device(dev_id)
            model = model.cuda(dev_id)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[dev_id])
            raise NotImplementedError
            # Not implemented, should divide data outside this function and get one portion here, probably with parallel version of `load_data`
    elif (dev_id is not None):
        if (type(dev_id) is list):
            model.cuda()
            model = DataParallel(model, device_ids=dev_id)
        else:
            torch.cuda.set_device(dev_id)
            model = model.cuda(dev_id)
    return model


def _step_gpt(model, optimizer, padtrim_X, padtrim_M, use_gpu=False):
    tkns_tnsr = torch.tensor(padtrim_X)
    mask_tnsr = torch.tensor(padtrim_M)
    try:
        if (use_gpu):
            tkns_tnsr = tkns_tnsr.to('cuda')
            mask_tnsr = mask_tnsr.to('cuda')
        loss = model(input_ids=tkns_tnsr[:,:-1], lm_labels=tkns_tnsr[:,1:])
        # if len(opts.devq) > 1: loss = loss.sum()
        loss = loss.view(-1, tkns_tnsr.size()[1] - 1)
        loss = ((loss * mask_tnsr[:, 1:]).sum(1) / mask_tnsr[:, 1:].sum(1)).mean()
        loss.backward()
        optimizer.step()
        del tkns_tnsr, mask_tnsr
    except:
        del tkns_tnsr, mask_tnsr
        torch.cuda.empty_cache()
        raise
    return loss.item()


def train(dev_id=None):
    use_gpu = dev_id is not None
    if opts.resume:
        print('Loading previously trained model...')
        checkpoint = torch.load(opts.resume)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        if ('resume' in checkpoint):
            print('Resuming from %i epoch(s) and %i sentence(s)' % (checkpoint['resume']['epoch']+1, checkpoint['resume']['sample']))
    else:
        model = MODEL_MAP[opts.model].from_pretrained(MODEL_NAME_MAP[opts.model])
        # try:
        # 	common_cfg, func_cfg = cfgr('train', 'common'), cfgr('train', 'train')
        # 	pr = io.param_reader(os.path.join(PAR_DIR, 'etc', '%s.yaml' % common_cfg.setdefault('mdl_cfg', 'mdlcfg')))
        # 	params = pr('LM', 'GPT-2')
        # 	config = CONFIG_MAP[opts.model](**params)
        # 	model = MODEL_MAP[opts.model](config)
        # except Exception as e:
        #     print(e)
        #     print('Cannot find the pretrained model file, using online model instead.')
        #     model = MODEL_MAP[opts.model].from_pretrained(MODEL_NAME_MAP[opts.model])
    if (use_gpu): model = _handle_model(model, dev_id=dev_id, distrb=opts.distrb)
    in_wrapper = opts.distrb or type(dev_id) is list
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    if (opts.resume): optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opts.epochs, eta_min=0, last_epoch=checkpoint['resume']['epoch']-1 if opts.resume and 'resume' in checkpoint else -1)
    model.train()

    tokenizer = TKNZR_MAP[opts.model].from_pretrained(MODEL_NAME_MAP[opts.model])
    # try:
    #     vocab_file = opts.vocab if os.path.isfile(opts.vocab) else func_cfg.setdefault('vocabulary', 'vocab.json')
    #     bpemerge_file = opts.bpe if os.path.isfile(opts.bpe) else func_cfg.setdefault('bpemerges', 'merges.txt')
    #     print('Using vocabulary file `%s` and bpe merge file `%s`' % (vocab_file, bpemerge_file))
    #     tokenizer = TKNZR_MAP[opts.model](vocab_file=vocab_file, merges_file=bpemerge_file, errors='replace', max_len=opts.maxlen)
    # except Exception as e:
    #     print(e)
    #     print('Cannot find the pretrained tokenizer file, using online model instead.')
    #     tokenizer = TKNZR_MAP[opts.model].from_pretrained(MODEL_NAME_MAP[opts.model])
    killer = system.GracefulKiller()
    for epoch in range(opts.epochs):
        if (opts.resume and 'resume' in checkpoint and checkpoint['resume']['epoch'] > epoch): continue
        t0 = time.time()
        batch_lastid, batch_tkns, sent_count, batch_count, total_loss = opts.bsize - 1, [], 0, 0, 0.
        data_gen = load_data(data_path=opts.input, model_name=opts.model, tokenizer=tokenizer, level='sent')
        if (opts.resume and 'resume' in checkpoint and checkpoint['resume']['epoch'] == epoch):
            elapsed_samples = int((checkpoint['resume']['sample'] / opts.bsize) * opts.bsize)
            for _ in range(elapsed_samples): next(data_gen)
            sent_count = elapsed_samples
            batch_count = int(sent_count / opts.bsize)
        for s in data_gen:
            batch_tkns.append(s)
            if (sent_count % opts.bsize == batch_lastid):
                padtrim_X, padtrim_M = _padtrim(batch_tkns, opts.maxlen + 1, pad_val=0)
                for _ in range(opts.maxtrial):
                    try:
                        total_loss += _step_gpt(model, optimizer, padtrim_X, padtrim_M, use_gpu=use_gpu)
                        break
                    except Exception as e:
                        print(e)
                else:
                    train_time = time.time() - t0
                    print('Interrupted! training time for %i epoch(s), %i batch(s) and %i sentence(s): %0.3fs' % (epoch + 1, batch_count, sent_count, train_time))
                    save_model(model, optimizer, '%s_checkpoint.pth' % opts.model, in_wrapper=in_wrapper, devq=opts.devq, resume={'epoch':epoch, 'sample':batch_count * opts.bsize})
                    sys.exit(0)
                batch_tkns = []
                batch_count += 1
                if (opts.autosave > 0 and batch_count % opts.autosave == 0):
                    print('Autosaving model...')
                    chkpt_fname ='%s_autosave_checkpoint.pth' % opts.model
                    save_model(model, optimizer, chkpt_fname, in_wrapper=in_wrapper, devq=opts.devq, resume={'epoch':epoch, 'sample':batch_count * opts.bsize})
                    del model
                    torch.cuda.empty_cache()
                    print('Loading autosaved model...')
                    checkpoint = torch.load(chkpt_fname)
                    model = checkpoint['model']
                    model.load_state_dict(checkpoint['state_dict'])
                    if (use_gpu): model = _handle_model(model, dev_id=dev_id, distrb=opts.distrb)
                    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opts.epochs, eta_min=0, last_epoch=checkpoint['resume']['epoch']-1)
            sent_count += 1
            if (killer.kill_now):
                train_time = time.time() - t0
                print('Interrupted! training time for %i epoch(s), %i batch(s) and %i sentence(s): %0.3fs' % (epoch + 1, batch_count, sent_count, train_time))
                save_model(model, optimizer, '%s_checkpoint.pth' % opts.model, in_wrapper=in_wrapper, devq=opts.devq, resume={'epoch':epoch, 'sample':batch_count * opts.bsize})
                sys.exit(0)
        if (len(batch_tkns) > 0):
            torch.cuda.empty_cache()
            padtrim_X, padtrim_M = _padtrim(batch_tkns, opts.maxlen + 1, pad_val=0)
            total_loss += _step_gpt(model, optimizer, padtrim_X, padtrim_M, use_gpu=use_gpu)
        scheduler.step()
        train_time = time.time() - t0
        print('Training time for %i epoch(s): %0.3fs' % (epoch + 1, train_time))

    save_model(model, optimizer, '%s_checkpoint.pth' % opts.model, opts.devq)


def train_bert():
    pass


def save_model(model, optimizer, fpath='checkpoint.pth', in_wrapper=False, devq=None, **kwargs):
    if in_wrapper: model = model.module
    model = model.cpu() if devq and len(devq) > 0 else model
    checkpoint = {'model': model, 'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict()}
    checkpoint.update(kwargs)
    torch.save(checkpoint, fpath)


def main():
    if (any(opts.model == mdl for mdl in ['gpt2', 'gpt'])):
        train_func = train
    elif (opts.model == 'bert'):
        train_func = train_bert
    if (opts.distrb):
        if (opts.np > 1): # Multi-process multiple GPU
            import torch.multiprocessing as mp
            mp.spawn(train_func, nprocs=len(opts.devq))
        else: # Single-process multiple GPU
            train_func(opts.devq if len(opts.devq) > 1 else opts.devq[0])
    elif (opts.devq): # Single-process
        train_func(opts.devq if len(opts.devq) > 1 else opts.devq[0])
    else:
        train_func() # CPU


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
    op.add_option('-j', '--epochs', default=1, action='store', type='int', dest='epochs', help='indicate the epoch used in deep learning')
    op.add_option('-z', '--bsize', default=32, action='store', type='int', dest='bsize', help='indicate the batch size used in deep learning')
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
    op.add_option('--seqlen', default=128, action='store', type='int', dest='seqlen', help='indicate the sequence length for each samples')
    op.add_option('--maxlen', default=128, action='store', type='int', dest='maxlen', help='indicate the maximum sequence length for each samples')
    op.add_option('--filter', default=0., action='store', type='float', dest='filter', help='threshold value')
    op.add_option('--resume', action='store', dest='resume', help='resume training file')
    op.add_option('--maxtrial', default=50, action='store', type='int', dest='maxtrial', help='maximum time to try')
    op.add_option('--autosave', default=10000, action='store', type='int', dest='autosave', help='indicate how frequently (#batches) to save model on disk')
    op.add_option('-i', '--input', help='input dataset')
    op.add_option('-m', '--model', default='gpt2', type='str', dest='model', help='the model to be validated')
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
    if (opts.distrb):
        import torch.distributed as dist
        dist.init_process_group(backend=opts.distbknd, init_method=args.disturl, world_size=opts.np, rank=opts.pid)

    main()
