#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: helper.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2019-03-25 19:10:10
###########################################################################
#

import os, sys, ast, time, logging
import argparse


global FILE_DIR, PAR_DIR, CONFIG_FILE, cfgr, args
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
args, cfgr = {}, None


def _convert_vocab(ifpath, ofpath):
    import json
    with open(ifpath, 'r') as f:
        line, vocab = f.readline(), {}
        while line:
            vocab.update(dict([tuple(line.split())]))
            line = f.readline()
        content = json.dumps(vocab)
    	# content = json.dumps(dict([tuple(line.split()) for line in f.readlines()]))
    with open(ofpath, 'w') as f:
        f.write(content)


def convert_vocab():
    _convert_vocab(ifpath=os.path.join(args.loc, args.input), ofpath=os.path.join(args.loc, args.output))


def _convert_vocabj(ifpath, ofpath):
    import json
    with open(ifpath, 'r') as f:
        vocab = json.load(f)
        content = '\n'.join(['%s %s' % (k, v) for k, v in vocab.iteritems()])
    with open(ofpath, 'w') as f:
        f.write(content)


def convert_vocabj():
    _convert_vocabj(ifpath=os.path.join(args.loc, args.input), ofpath=os.path.join(args.loc, args.output))


def merge_vocab():
    import json, operator
    vocab_fpaths = [os.path.join(args.loc, fname) for fname in args.input.split(args.sep)]
    merged_vocab = {}
    for vocabf in vocab_fpaths:
        try:
            fname, fext = os.path.splitext(vocabf)
            if (fext == 'txt'):
                new_fpath = fname + '.json'
                _convert_vocab(vocabf, new_fpath)
                vocabf = new_fpath
            with open(vocabf, 'r') as f:
                vocab = json.load(f)
                for k, v in vocab.iteritems():
                    merged_vocab.setdefault(k, 0)
                    merged_vocab[k] += int(v)
        except Exception as e:
            print(e)
    vocab_data = json.dumps(merged_vocab)
    with open(args.output if args.output else 'merged_vocab.json', 'w') as f:
        f.write(vocab_data)
    with open(args.output if args.output else 'merged_vocab.txt', 'w') as f:
        f.write('\n'.join(['%s %s' % (k.encode('utf-8'), v) for k, v in sorted(merged_vocab.items(), key=operator.itemgetter(1), reverse=True)]))


def merge_bpe():
    from collections import OrderedDict
    bpe_fpaths = [os.path.join(args.loc, fname) for fname in args.input.split(args.sep)]
    merged_bpe = []
    for bpef in bpe_fpaths:
        try:
            with open(bpef, 'r') as f:
                merged_bpe.extend(f.readlines())
        except Exception as e:
            print(e)
    bpe_data = ''.join(list(OrderedDict.fromkeys(merged_bpe)))
    with open(args.output if args.output else 'merged_bpe.txt', 'w') as f:
        f.write(bpe_data)


def decouple_config():
    sys.path.insert(0, PAR_DIR)
    from util.config import SimpleConfig
    config_fpath = os.path.join(args.loc, args.input)
    pkl_fpath = args.cfg.setdefault('pkl_fpath', os.path.splitext(config_fpath)[0]+'.pkl')
    keep_obj = args.cfg.setdefault('keepobj', False)
    skip_paths = args.cfg.setdefault('skippaths', [])
    config = SimpleConfig.from_file(config_fpath, pkl_fpath=pkl_fpath, decouple=True, keep_obj=keep_obj, skip_paths=skip_paths)
    config.to_file(args.input)
    config.output_importmap()
    print(SimpleConfig.from_file(config_fpath, pkl_fpath=pkl_fpath, decouple=True, keep_obj=keep_obj, import_lib=True, skip_paths=skip_paths).__dict__)


def main():
    if (args.method == 'cnvrt-vcb'):
        convert_vocab()
    if (args.method == 'cnvrt-vcbj'):
        convert_vocabj()
    elif (args.method == 'mrg-vcb'):
        merge_vocab()
    elif (args.method == 'mrg-bpe'):
        merge_bpe()
    elif (args.method == 'decpl-cfg'):
        decouple_config()


if __name__ == '__main__':
    # Logging setting
    logging.basicConfig(level=logging.INFO, format='%(aSEPtime)s %(levelname)s %(message)s')

    # Parse commandline arguments
    parser = argparse.ArgumentParser(description='Helper script.')
    parser.add_argument('-l', '--loc', default='.', help='the files in which location to be process')
    parser.add_argument('-i', '--input', help='input file')
    parser.add_argument('-o', '--output', help='output file')
    parser.add_argument('--sep', default=';;', help='the separator in the string')
    parser.add_argument('-c', '--cfg', help='config string used to update the settings, format: {\'param_name1\':param_value1[, \'param_name1\':param_value1]}')
    parser.add_argument('-m', '--method', help='main method to run')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='display detailed information')
    args = parser.parse_args()

    cfg_kwargs = {} if args.cfg is None else ast.literal_eval(args.cfg)
    args.cfg = cfg_kwargs

    main()
