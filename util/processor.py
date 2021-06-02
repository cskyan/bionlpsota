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

import os, sys, pickle, itertools, logging

import numpy as np

import torch

import ftfy, spacy
try:
    nlp = spacy.load('en_core_sci_md')
except Exception as e:
    logging.warning(e)
    try:
        nlp = spacy.load('en_core_sci_sm')
    except Exception as e:
        logging.warning(e)
        nlp = spacy.load('en_core_web_sm')

from bionlp.nlp import AdvancedCountVectorizer, AdvancedTfidfVectorizer
from bionlp.util import math as imath


def _base_transform(sample, input_keys=[]):
    X, y = sample
    X = [X[k] for k in input_keys]
    if len(X) == 0: raise Exception('No inputs generated!')
    return X, y


def _bert_input_keys(task_type=None):
    return ['input_ids', 'attention_mask'] + ([] if task_type is None or task_type not in ['entlmnt', 'sentsim'] else ['token_type_ids'])


def _gpt_input_keys(task_type=None):
    return ['input_ids', 'attention_mask']


def _embed_input_keys(embed_type='w2v'):
    embed_types = embed_type.split('_')
    return ['input_ids', 'mask'] + (embed_types[1:] if len(embed_types) > 1 else [])


def _nn_transform(sample, **kwargs):
    maxlen = kwargs.setdefault('maxlen', 128)
    pad_token = kwargs.setdefault('pad_token', '<PAD>')
    X, y = sample
    if (type(X[0]) is not str and hasattr(X[0], '__iter__')):
        pad_idx = [len(x) for x in X]
        for x in range(len(X)):
            try:
                pad_idx[x] = X[x].index(pad_token)
            except ValueError as e:
                pass
        mask = [[1] * pad_idx[x] + [0] * (len(X[x]) - pad_idx[x]) for x in range(len(X))]
    else:
        try:
            pad_idx = X.index(pad_token)
        except ValueError as e:
            pad_idx = len(X)
        mask = [1] * pad_idx + [0] * (len(X) - pad_idx)
    mask = mask[:min(maxlen, len(mask))] + [0] * max(0, maxlen-len(mask))
    return [X, mask], y


def _w2v_transform(sample, **kwargs):
    maxlen = kwargs.setdefault('maxlen', 128)
    kwargs['pad_token'] = -1
    X, y = sample
    from gensim.models import Word2Vec
    w2v_model = kwargs['w2v_model'] if 'w2v_model' in kwargs and kwargs['w2v_model'] else Word2Vec.load("word2vec.model")
    X = [[w2v_model.vocab[w].index if w in w2v_model.vocab else (w2v_model.vocab[w.lower()].index if w.lower() in w2v_model.vocab else 0) for w in X[x]] for x in range(len(X))] if type(X[0]) is not str else [w2v_model.vocab[w].index if w in w2v_model.vocab else (w2v_model.vocab[w.lower()].index if w.lower() in w2v_model.vocab else 0) for w in X]
    X = _padtrim(X, maxlen=maxlen, pad_token=kwargs['pad_token'])
    return _nn_transform([X, y], **kwargs)


def _elmo_transform(sample, **kwargs):
    maxlen = kwargs.setdefault('maxlen', 128)
    pad_token = kwargs.setdefault('pad_token', '<PAD>')
    lm_config = kwargs.setdefault('lm_config', {})
    num_layers = lm_config.setdefault('num_output_representations', 2)
    X, y = sample
    X = _trimming(X, maxlen=maxlen)
    dummy_sent = [pad_token] * maxlen
    from allennlp.modules.elmo import batch_to_ids
    _X = batch_to_ids(X+[dummy_sent]).tolist() if type(X[0]) is not str else batch_to_ids([X, dummy_sent]).tolist()
    X = _X[:-1] if len(_X) > 2 else _X[0]
    kwargs['pad_token'] = _X[-1][0][-1] if num_layers > 1 else _X[-1][-1]
    if num_layers > 1:
        if len(_X) > 2:
            (_, mask), y = _nn_transform([[X[x][0] for x in range(len(X))], y], **kwargs)
        else:
            (_, mask), y = _nn_transform([X[0], y], **kwargs)
    else:
        (_, mask), y = _nn_transform([X, y], **kwargs)
    return [X, mask], y


def _sentvec_transform(sample, **kwargs):
    X, y = sample
    import sent2vec
    sentvec_model = kwargs['sentvec_model'] if 'sentvec_model' in kwargs and kwargs['sentvec_model'] else sent2vec.Sent2vecModel()
    X = sentvec_model.embed_sentences([' '.join(x) for x in X]) if type(X[0]) is not str else sentvec_model.embed_sentences([' '.join(X)])[0]
    mask = [1]*len(X) if type(X[0]) is not str else [[1]*len(X[x]) for x in range(len(X))]
    return [X, mask], y


def _embedding_transform(sample, **kwargs):
    embed_types = kwargs.setdefault('embed_type', 'w2v').split('_')
    embedding_ids, masks, y = [], [], 0
    for embed_t in embed_types:
        X, y = EMBED_TRANSFORM[embed_t](sample, **kwargs)
        embedding_ids.append(X[0])
        masks.append(X[1])
    inputs = dict(zip(embed_types, embedding_ids))
    X = inputs[embed_types[0]]
    return [X, masks[0]]+[inputs[x] for x in embed_types[1:]], y


EMBED_TRANSFORM = {'w2v': _w2v_transform, 'sentvec': _sentvec_transform}


def _adjust_encoder(tokenizer, config):
    encoded_extknids = []
    if (config.model.startswith('bert')):
        pass
    elif (config.model == 'gpt'):
        tokenizer.cls_token, tokenizer.eos_token, tokenizer.pad_token = '<CLS>', '<EOS>', '<PAD>'
    elif (config.model == 'gpt2'):
        tokenizer.pad_token = tokenizer.eos_token
    elif (config.model == 'trsfmxl'):
        pass
    elif (hasattr(config, 'embed_type') and config.model in config.embed_type):
        tokenizer.update({'bos_token':'<BOS>', 'eos_token':'<EOS>', 'pad_token':'<PAD>'})
    else:
        pass


def _base_encode(text, tokenizer, tknz_kwargs={}):
    try:
        try:
            record = tokenizer(text, **tknz_kwargs) if (type(text) is str or not hasattr(text, '__iter__')) else tokenizer(*text, **tknz_kwargs)
        except Exception as e:
            record = tokenizer(ftfy.fix_text(text), **tknz_kwargs) if (type(text) is str or not hasattr(text, '__iter__')) else tokenizer(*[ftfy.fix_text(str(s)) for s in text], **tknz_kwargs)
        return record
    except Exception as e:
        logging.warning(e)
        logging.warning('Cannot encode %s' % str(text).encode('ascii', 'replace').decode('ascii'))
        return {}


def _txt2vec(texts, config, clf_h=None, txt_vctrz=None, char_vctrz=None, use_tfidf=True, ftdecomp=None, ftmdl=None, n_components=128, saved_path='.', prefix='corpus', **kwargs):
    extra_outputs = ()
    logging.info('Converting text to vectors with parameters: %s; %s' % (str(dict(txt_vctrz=txt_vctrz, char_vctrz=char_vctrz, use_tfidf=use_tfidf, ftdecomp=ftdecomp, ftmdl=ftmdl, n_components=n_components, prefix=prefix, saved_path=saved_path)), str(kwargs)))
    from scipy.sparse import csr_matrix, hstack, issparse
    if config.sentvec_path and os.path.isfile(config.sentvec_path):
        import sent2vec
        sentvec_model = sent2vec.Sent2vecModel()
        sentvec_model.load_model(config.sentvec_path)
        sentvec = sentvec_model.embed_sentences(texts)
        logging.info('Sentence vector dimension of dataset %s: %i' % (prefix, sentvec.shape[1]))
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
                logging.info('Eval mode of TFIDF:')
                txt_X = txt_vctrz.transform(texts)
            with open('%s_tfidf.pkl' % prefix, 'wb') as fd:
                pickle.dump((txt_X, txt_vctrz), fd)
        logging.info('TFIDF dimension of dataset %s: %i' % (prefix, txt_X.shape[1]))
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
                logging.info('Eval mode of Char TFIDF:')
                char_X = char_vctrz.transform(texts)
            with open('%s_chartfidf.pkl' % prefix, 'wb') as fd:
                pickle.dump((char_X, char_vctrz), fd)
        logging.info('Char TFIDF dimension of dataset %s: %i' % (prefix, char_X.shape[1]))
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
        logging.info('BM25 dimension of dataset %s: %i' % (prefix, txt_bm25_X.shape[1]))
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
        logging.info('Using %s feature reduction...' % ftdecomp.upper())
        clf_h = ftmdl.fit_transform(clf_h).astype('float32')
    else:
        logging.info('Eval mode of feature reduction:')
        clf_h = ftmdl.transform(clf_h).astype('float32')
    return (clf_h, txt_vctrz, char_vctrz, ftmdl) + extra_outputs


def _tokenize(text, tokenizer, tknz_kwargs={}):
    try:
        records = [w.text for w in nlp(ftfy.fix_text(text))] if (type(text) is str) else [[w.text for w in nlp(ftfy.fix_text(line))] for line in text]
    except Exception as e:
        logging.warning(e)
        logging.warning('Cannot encode %s' % str(text.encode('ascii', 'replace').decode('ascii')))
        return []
    return records


def _dummy_tokenize(text, tokenizer, tknz_kwargs={}):
    return text


def _trimming(tokenized_text, maxlen=128):
    return tokenized_text[:min(len(tokenized_text), maxlen)] if (type(tokenized_text[0]) is str or type(tokenized_text[0]) is int) else [s[:min(len(s), maxlen)] for s in tokenized_text]


def _padding(tokenized_text, maxlen=128, pad_token='<PAD>'):
    # pad_token = config.tknzr.setdefault('pad_token', '<PAD>') if type(config.tknzr) is dict else config.tknzr.pad_token
    return (tokenized_text + [pad_token]*(maxlen-len(tokenized_text))) if (type(tokenized_text[0]) is str or type(tokenized_text[0]) is int) else [s + [pad_token]*(maxlen-len(s)) for s in tokenized_text]


def _padtrim(tokenized_text, maxlen=128, pad_token='<PAD>'):
    return (tokenized_text[:min(len(tokenized_text), maxlen)] + [pad_token]*(maxlen-len(tokenized_text))) if (type(tokenized_text[0]) is str or type(tokenized_text[0]) is int) else [s[:min(len(s), maxlen)] + [pad_token]*(maxlen-len(s)) for s in tokenized_text]


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
