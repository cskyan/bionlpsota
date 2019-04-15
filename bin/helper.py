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

import os, sys, time, logging
from optparse import OptionParser


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PAR_DIR = os.path.abspath(os.path.join(FILE_DIR, os.path.pardir))
CONFIG_FILE = os.path.join(PAR_DIR, 'etc', 'config.yaml')
SEP=';;'

opts, args = {}, []
cfgr = None


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
    _convert_vocab(ifpath=os.path.join(opts.loc, opts.input), ofpath=os.path.join(opts.loc, opts.output))


def _convert_vocabj(ifpath, ofpath):
    import json
    with open(ifpath, 'r') as f:
        vocab = json.load(f)
        content = '\n'.join(['%s %s' % (k, v) for k, v in vocab.iteritems()])
    with open(ofpath, 'w') as f:
        f.write(content)


def convert_vocabj():
    _convert_vocabj(ifpath=os.path.join(opts.loc, opts.input), ofpath=os.path.join(opts.loc, opts.output))


def merge_vocab():
    import json, operator
    vocab_fpaths = [os.path.join(opts.loc, fname) for fname in opts.input.split(SEP)]
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
    with open(opts.output if opts.output else 'merged_vocab.json', 'w') as f:
        f.write(vocab_data)
    with open(opts.output if opts.output else 'merged_vocab.txt', 'w') as f:
        f.write('\n'.join(['%s %s' % (k.encode('utf-8'), v) for k, v in sorted(merged_vocab.items(), key=operator.itemgetter(1), reverse=True)]))


def merge_bpe():
    from collections import OrderedDict
    bpe_fpaths = [os.path.join(opts.loc, fname) for fname in opts.input.split(SEP)]
    merged_bpe = []
    for bpef in bpe_fpaths:
        try:
            with open(bpef, 'r') as f:
                merged_bpe.extend(f.readlines())
        except Exception as e:
            print(e)
    bpe_data = ''.join(list(OrderedDict.fromkeys(merged_bpe)))
    with open(opts.output if opts.output else 'merged_bpe.txt', 'w') as f:
        f.write(bpe_data)


def main():
    if (opts.method == 'cnvrt-vcb'):
        convert_vocab()
    elif (opts.method == 'mrg-vcb'):
        merge_vocab()
    elif (opts.method == 'mrg-bpe'):
        merge_bpe()


if __name__ == '__main__':
    # Logging setting
    logging.basicConfig(level=logging.INFO, format='%(aSEPtime)s %(levelname)s %(message)s')

    # Parse commandline arguments
    op = OptionParser()
    op.add_option('-l', '--loc', default='.', help='the files in which location to be process')
    op.add_option('-i', '--input', help='input file')
    op.add_option('-o', '--output', help='output file')
    op.add_option('--sep', help='the separator in the string')
    op.add_option('-m', '--method', help='main method to run')
    op.add_option('-v', '--verbose', action='store_true', dest='verbose', default=False, help='display detailed information')

    (opts, args) = op.parse_args()
    if len(args) > 0:
    	op.print_help()
    	op.error('Please input options instead of arguments.')
    	sys.exit(1)
    if opts.sep: SEP=sep

    main()
