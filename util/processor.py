#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2021 by Caspar. All rights reserved.
# File Name: processor.py
# Author: Shankai Yan
# E-mail: dr.skyan@gmail.com
# Created Time: 2021-03-29 20:25:25
###########################################################################
#

import os, sys, pickle, itertools

import numpy as np

import torch

import ftfy, spacy
try:
    nlp = spacy.load('en_core_sci_md')
except Exception as e:
    print(e)
    try:
        nlp = spacy.load('en_core_sci_sm')
    except Exception as e:
        print(e)
        nlp = spacy.load('en_core_web_sm')

from bionlp.nlp import AdvancedCountVectorizer, AdvancedTfidfVectorizer
from bionlp.util import math as imath


def _sentclf_transform(sample, options=None, model=None, seqlen=32, start_tknids=[], clf_tknids=[], **kwargs):
    X, y = sample
    if model == 'bert' and (kwargs.setdefault('sentsim_func', None) is None or kwargs['sentsim_func']=='concat'):
        X = [start_tknids + x for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else start_tknids + X
    else: # GPT
        X = [start_tknids + x + clf_tknids for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else start_tknids + X + clf_tknids
    return X, y


def _entlmnt_transform(sample, options=None, model=None, seqlen=32, start_tknids=[], clf_tknids=[], delim_tknids=[], **kwargs):
    X, y = sample
    if model == 'bert':
        if kwargs.setdefault('sentsim_func', None) is None:
            trim_len = int(np.ceil((sum([len(v) for v in [start_tknids, X[0], delim_tknids, X[1], delim_tknids]]) - seqlen) / 2.0))
            X = [x[:len(x)-trim_len] for x in X]
            X = start_tknids + X[0] + delim_tknids + X[1] + delim_tknids
        else:
            pass
    else: # GPT
        if kwargs.setdefault('sentsim_func', None) is None:
            trim_len = int(np.ceil((sum([len(v) for v in [start_tknids, X[0], delim_tknids, X[1], clf_tknids]]) - seqlen) / 2.0))
            X = [x[:len(x)-trim_len] for x in X]
            X = start_tknids + X[0] + delim_tknids + X[1] + clf_tknids
        else:
            pass
    return X, y


def _sentsim_transform(sample, options=None, model=None, seqlen=32, start_tknids=[], clf_tknids=[], delim_tknids=[], **kwargs):
    X, y = sample
    if model == 'bert':
        if kwargs.setdefault('sentsim_func', None) is None:
            trim_len = int(np.ceil((sum([len(v) for v in [start_tknids, X[0], delim_tknids, X[1], delim_tknids]]) - seqlen) / 2.0))
            X = [x[:len(x)-trim_len] for x in X]
            X = start_tknids + X[0] + delim_tknids + X[1] + delim_tknids
        else:
            pass
    else: # GPT
        trim_len = int(np.ceil((sum([len(v) for v in [start_tknids, X[0], delim_tknids, X[1], clf_tknids]]) - seqlen) / 2.0))
        X = [x[:len(x)-trim_len] for x in X]
        X = [start_tknids + X[0] + delim_tknids + X[1] + clf_tknids, start_tknids + X[1] + delim_tknids + X[0] + clf_tknids]
    return X, y


def _padtrim_transform(sample, options=None, seqlen=32, xpad_val=0, ypad_val=None, **kwargs):
    X, y = sample
    X = [x[:min(seqlen, len(x))] + [xpad_val] * (seqlen - len(x)) for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else X[:min(seqlen, len(X))] + [xpad_val] * (seqlen - len(X))
    num_trim_delta = len([1 for x in X if seqlen > len(x)]) if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else 1 if seqlen > len(X) else 0
    # if num_trim_delta > 0:
    #     global NUM_TRIM
    #     NUM_TRIM += num_trim_delta
    #     if NUM_TRIM % 100 == 0: print('Triming too much sentences! Please consider using a larger maxlen parameter!')
    if ypad_val is not None: y = [x[:min(seqlen, len(x))] + [ypad_val] * (seqlen - len(x)) for x in y] if hasattr(y, '__iter__') and len(y) > 0 and type(y[0]) is not str and hasattr(y[0], '__iter__') else y[:min(seqlen, len(y))] + [ypad_val] * (seqlen - len(y))
    return X, y


def _trim_transform(sample, options=None, seqlen=32, trimlbs=False, required_special_tkns=[], special_tkns={}, **kwargs):
    seqlen -= sum([len(v) for v in special_tkns.values()])
    X, y = sample
    X = [x[:min(seqlen, len(x))] for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else X[:min(seqlen, len(X))]
    # num_trim_delta = len([1 for x in X if seqlen > len(x)]) if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else 1 if seqlen > len(X) else 0
    # if num_trim_delta > 0:
    #     global NUM_TRIM
    #     NUM_TRIM += num_trim_delta
    #     if NUM_TRIM % 100 == 0: print('Triming too much sentences! Please consider using a larger maxlen parameter!')
    if trimlbs: y = [x[:min(seqlen, len(x))] for x in y] if hasattr(y, '__iter__') and len(y) > 0 and type(y[0]) is not str and hasattr(y[0], '__iter__') else y[:min(seqlen, len(y))]
    return X, y


def _pad_transform(sample, options=None, seqlen=32, xpad_val=0, ypad_val=None, **kwargs):
    X, y = sample
    X = [x + [xpad_val] * (seqlen - len(x)) for x in X] if hasattr(X, '__iter__') and len(X) > 0 and type(X[0]) is not str and hasattr(X[0], '__iter__') else X + [xpad_val] * (seqlen - len(X))
    if ypad_val is not None: y = [x + [ypad_val] * (seqlen - len(x)) for x in y] if hasattr(y, '__iter__') and len(y) > 0 and type(y[0]) is not str and hasattr(y[0], '__iter__') else y + [ypad_val] * (seqlen - len(y))
    return X, y


def _dummy_trsfm(sample, **kwargs):
    return sample


def _adjust_encoder(mdl_name, tokenizer, config, extra_tokens=[], ret_list=False):
    encoded_extknids = []
    if (mdl_name.startswith('bert')):
        for tkn in extra_tokens:
            tkn_ids = tokenizer.tokenize(tkn)
            encoded_extknids.append([tokenizer.convert_tokens_to_ids(tkn_ids)] if (ret_list and type(tkn_ids) is not list) else tokenizer.convert_tokens_to_ids(tkn_ids))
    elif (mdl_name == 'gpt'):
        for tkn in extra_tokens:
            # tokenizer.encoder[tkn] = len(tokenizer.encoder)
            # encoded_extknids.append([tokenizer.encoder[tkn]] if ret_list else tokenizer.encoder[tkn])
            encoded_extknids.append([tokenizer.convert_tokens_to_ids(tkn)] if ret_list else tokenizer.convert_tokens_to_ids(tkn))
    elif (mdl_name == 'gpt2'):
        encoded_extknids = []
        for tkn in extra_tokens:
            tkn_ids = tokenizer.encode(tkn)
            encoded_extknids.append([tkn_ids] if (ret_list and type(tkn_ids) is not list) else tkn_ids)
    elif (mdl_name == 'trsfmxl'):
        for tkn in extra_tokens:
            tokenizer.__dict__[tkn] = len(tokenizer.__dict__)
            encoded_extknids.append([tokenizer.__dict__[tkn]] if ret_list else tokenizer.__dict__[tkn])
    elif (hasattr(config, 'embed_type') and mdl_name in config.embed_type):
        encoded_extknids = [[tkn] if ret_list else tkn for tkn in extra_tokens]
    else:
        encoded_extknids = [None] * len(extra_tokens)
    return encoded_extknids


def _base_encode(text, tokenizer):
    texts, records = [str(text)] if (type(text) is str or not hasattr(text, '__iter__')) else [str(s) for s in text], []
    try:
        for txt in texts:
            tokens = tokenizer.tokenize(ftfy.fix_text(txt))
            record = []
            while (len(tokens) > 512):
               record.extend(tokenizer.convert_tokens_to_ids(tokens[:512]))
               tokens = tokens[512:]
            record.extend(tokenizer.convert_tokens_to_ids(tokens))
            records.append(record)
    except Exception as e:
        print(e)
        print('Cannot encode %s' % str(text).encode('ascii', 'replace').decode('ascii'))
        return []
    return records[0] if (type(text) is str or not hasattr(text, '__iter__')) else records


def _gpt2_encode(text, tokenizer):
    try:
        records = tokenizer.encode(ftfy.fix_text(str(text)).encode('ascii', 'replace').decode('ascii')) if (type(text) is str or not hasattr(text, '__iter__')) else [tokenizer.encode(ftfy.fix_text(str(line)).encode('ascii', 'replace').decode('ascii')) for line in text]
    except ValueError as e:
        try:
            records = list(itertools.chain(*[tokenizer.encode(w.text) for w in nlp(ftfy.fix_text(str(text)))])) if (type(text) is str or not hasattr(text, '__iter__')) else list(itertools.chain(*[list(itertools.chain(*[tokenizer.encode(w.text) for w in nlp(ftfy.fix_text(str(line)))])) for line in text]))
        except Exception as e:
            print(e)
            print('Cannot encode %s' % str(text.encode('ascii', 'replace').decode('ascii')))
            return []
    except Exception as e:
        print(e)
        print('Cannot encode %s' % str(text.encode('ascii', 'replace').decode('ascii')))
        return []
    return records


def _txt2vec(texts, config, clf_h=None, txt_vctrz=None, char_vctrz=None, use_tfidf=True, ftdecomp=None, ftmdl=None, n_components=128, saved_path='.', prefix='corpus', **kwargs):
    extra_outputs = ()
    print('Converting text to vectors with parameters: %s; %s' % (str(dict(txt_vctrz=txt_vctrz, char_vctrz=char_vctrz, use_tfidf=use_tfidf, ftdecomp=ftdecomp, ftmdl=ftmdl, n_components=n_components, prefix=prefix, saved_path=saved_path)), str(kwargs)))
    from scipy.sparse import csr_matrix, hstack, issparse
    if config.sentvec_path and os.path.isfile(config.sentvec_path):
        import sent2vec
        sentvec_model = sent2vec.Sent2vecModel()
        sentvec_model.load_model(config.sentvec_path)
        sentvec = sentvec_model.embed_sentences(texts)
        print('Sentence vector dimension of dataset %s: %i' % (prefix, sentvec.shape[1]))
        clf_h = hstack((csr_matrix(clf_h), txt_X)) if clf_h is not None else sentvec
    if config.do_tfidf:
        tfidf_cache_fpath = os.path.join(saved_path, '%s_tfidf.pkl' % prefix)
        if os.path.exists(tfidf_cache_fpath):
            with open(tfidf_cache_fpath, 'rb') as fd:
                txt_X, txt_vctrz = pickle.load(fd)
        else:
            if txt_vctrz is None:
                binary = (ftdecomp=='svd' or kwargs.setdefault('binary', False))
                txt_vctrz = AdvancedTfidfVectorizer(stop_words=kwargs.setdefault('stop_words', 'english'), ngram_range=kwargs.setdefault('ngram', (1,1)), binary=binary, dtype='float32', use_idf=kwargs.setdefault('use_idf', True), sublinear_tf=kwargs.setdefault('sublinear_tf', False), lemma=kwargs.setdefault('lemma', False), stem=kwargs.setdefault('stem', False), synonym=kwargs.setdefault('synonym', False), w2v_fpath=kwargs.setdefault('w2v_fpath', None), w2v_topk=kwargs.setdefault('w2v_topk', 10), phraser_fpath=kwargs.setdefault('phraser', None), keep_orig=kwargs.setdefault('keep_orig', False)) if use_tfidf else AdvancedCountVectorizer(stop_words=kwargs.setdefault('stop_words', 'english'), ngram_range=kwargs.setdefault('ngram', (1,1)), binary=binary, dtype='int8' if binary else 'int32', lemma=kwargs.setdefault('lemma', False), stem=kwargs.setdefault('stem', False), synonym=kwargs.setdefault('synonym', False), w2v_fpath=kwargs.setdefault('w2v_fpath', None), w2v_topk=kwargs.setdefault('w2v_topk', 10), phraser_fpath=kwargs.setdefault('phraser', None), keep_orig=kwargs.setdefault('keep_orig', False))
                txt_X = txt_vctrz.fit_transform(texts)
                if len(kwargs.setdefault('ngram_weights', {})) == txt_vctrz.get_params()['ngram_range'][1] - txt_vctrz.get_params()['ngram_range'][0] + 1:
                    ngram_types = np.array(list(map(lambda x: x.count(' ')+1, txt_vctrz.get_feature_names())))
                    ngram_idx = dict((tp, np.where(ngram_types == tp)[0]) for tp in np.unique(ngram_types))
                    if all([k in kwargs['ngram_weights'] for k in ngram_idx.keys()]):
                        norm_weights = imath.normalize(list(kwargs['ngram_weights'].values()))
                        for i, k in enumerate(kwargs['ngram_weights'].keys()): ngram_idx[k] = (ngram_idx[k], norm_weights[i])
                        extra_outputs += (ngram_idx,)
            else:
                print('Eval mode of TFIDF:')
                txt_X = txt_vctrz.transform(texts)
            with open('%s_tfidf.pkl' % prefix, 'wb') as fd:
                pickle.dump((txt_X, txt_vctrz), fd)
        print('TFIDF dimension of dataset %s: %i' % (prefix, txt_X.shape[1]))
        clf_h = hstack((csr_matrix(clf_h), txt_X)) if clf_h is not None else txt_X
    if config.do_chartfidf:
        chartfidf_cache_fpath = os.path.join(saved_path, '%s_chartfidf.pkl' % prefix)
        if os.path.exists(chartfidf_cache_fpath):
            with open(chartfidf_cache_fpath, 'rb') as fd:
                char_X, char_vctrz = pickle.load(fd)
        else:
            if char_vctrz is None:
                binary = (ftdecomp=='svd' or kwargs.setdefault('binary', False))
                char_vctrz = AdvancedTfidfVectorizer(analyzer=kwargs.setdefault('char_analyzer', 'char_wb'), stop_words=kwargs.setdefault('stop_words', 'english'), ngram_range=kwargs.setdefault('char_ngram', (4,6)), binary=binary, dtype='float32', use_idf=kwargs.setdefault('use_idf', True), sublinear_tf=kwargs.setdefault('sublinear_tf', False)) if use_tfidf else AdvancedCountVectorizer(analyzer=kwargs.setdefault('char_analyzer', 'char_wb'), stop_words=kwargs.setdefault('stop_words', 'english'), ngram_range=kwargs.setdefault('char_ngram', (4,6)), binary=binary, dtype='int8' if binary else 'int32')
                char_X = char_vctrz.fit_transform(texts)
                if len(kwargs.setdefault('ngram_weights', {})) == char_vctrz.get_params()['ngram_range'][1] - char_vctrz.get_params()['ngram_range'][0] + 1:
                    ngram_types = np.array(list(map(lambda x: x.count(' '), char_vctrz.get_feature_names())))
                    ngram_idx = dict((tp, np.where(ngram_types == tp)[0]) for tp in np.unique(ngram_types))
                    if all([k in kwargs['ngram_weights'] for k in ngram_idx.keys()]):
                        norm_weights = imath.normalize(kwargs['ngram_weights'].values())
                        for i, k in enumerate(kwargs['ngram_weights'].keys()): ngram_idx[k] = (ngram_idx[k], norm_weights[i])
                        extra_outputs += (ngram_idx,)
            else:
                print('Eval mode of Char TFIDF:')
                char_X = char_vctrz.transform(texts)
            with open('%s_chartfidf.pkl' % prefix, 'wb') as fd:
                pickle.dump((char_X, char_vctrz), fd)
        print('Char TFIDF dimension of dataset %s: %i' % (prefix, char_X.shape[1]))
        clf_h = hstack((csr_matrix(clf_h), char_X)) if clf_h is not None else char_X
    if config.do_bm25:
        bm25_cache_fpath = os.path.join(saved_path, '%s_bm25.pkl' % prefix)
        if os.path.exists(bm25_cache_fpath):
            with open(bm25_cache_fpath, 'rb') as fd:
                txt_bm25_X = pickle.load(fd)
        else:
            from gensim.summarization.bm25 import get_bm25_weights
            txt_bm25_X = np.array(get_bm25_weights(texts, n_jobs=config.np))
            with open('%s_bm25.pkl' % prefix, 'wb') as fd:
                pickle.dump(txt_bm25_X, fd)
        print('BM25 dimension of dataset %s: %i' % (prefix, txt_bm25_X.shape[1]))
        clf_h = hstack((csr_matrix(clf_h), txt_bm25_X)) if clf_h is not None else txt_bm25_X
    if type(ftdecomp) is str: ftdecomp = ftdecomp.lower()
    if issparse(clf_h) and ftdecomp != 'svd': clf_h = clf_h.toarray()
    # Feature reduction
    if ftdecomp is None or type(ftdecomp) is str and ftdecomp.lower() == 'none' or n_components >= clf_h.shape[1]: return clf_h, txt_vctrz, char_vctrz, None
    if ftmdl is None:
        if ftdecomp == 'pca':
            from sklearn.decomposition import PCA
            ftmdl = PCA(n_components=min(n_components, clf_h.shape[0]))
        elif ftdecomp == 'svd':
            from sklearn.decomposition import TruncatedSVD
            ftmdl = TruncatedSVD(n_components=n_components)
        print('Using %s feature reduction...' % ftdecomp.upper())
        clf_h = ftmdl.fit_transform(clf_h).astype('float32')
    else:
        print('Eval mode of feature reduction:')
        clf_h = ftmdl.transform(clf_h).astype('float32')
    return (clf_h, txt_vctrz, char_vctrz, ftmdl) + extra_outputs


def _tokenize(text, tokenizer):
    return text
    try:
        records = [w.text for w in nlp(ftfy.fix_text(text))] if (type(text) is str) else [[w.text for w in nlp(ftfy.fix_text(line))] for line in text]
    except Exception as e:
        print(e)
        print('Cannot encode %s' % str(text.encode('ascii', 'replace').decode('ascii')))
        return []
    return records


def _batch2ids_w2v(batch_text, w2v_model):
    return [[w2v_model.vocab[w].index if w in w2v_model.vocab else (w2v_model.vocab[w.lower()].index if w.lower() in w2v_model.vocab else 0) for w in line] for line in batch_text]


def _batch2ids_sentvec(batch_text, sentvec_model):
    return torch.tensor(sentvec_model.embed_sentences([' '.join(x) for x in batch_text]))


def _onehot(y, size):
    y = torch.LongTensor(y).view(-1, 1)
    y_onehot = torch.FloatTensor(size[0], size[1])
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot.long()
