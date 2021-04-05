#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2021 by Caspar. All rights reserved.
# File Name: config.py
# Author: Shankai Yan
# E-mail: dr.skyan@gmail.com
# Created Time: 2021-03-29 17:19:30
###########################################################################
#

from collections import OrderedDict

import numpy as np

import torch
from torch import nn

from allennlp.modules.elmo import Elmo
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
from allennlp.modules.seq2seq_encoders import FeedForwardEncoder, PytorchSeq2SeqWrapper, GatedCnnEncoder, IntraSentenceAttentionEncoder, QaNetEncoder, StackedSelfAttentionEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, PytorchSeq2VecWrapper, Seq2VecEncoder, CnnEncoder, CnnHighwayEncoder

from transformers import BertConfig, BertTokenizer, BertModel, AdamW, OpenAIGPTConfig, OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, TransfoXLConfig, TransfoXLTokenizer, TransfoXLLMHeadModel

from dataset import BaseDataset, SentSimDataset, EntlmntDataset, NERDataset
from processor import *
from ..modules.shell import BaseShell
from ..modules.constraint import HrchConstraint
from ..modules.embedding import EmbeddingPool, EmbeddingSeq2Vec, EmbeddingSeq2Seq, SentVecEmbeddingSeq2Vec
from ..modules.loss import MSELoss, ContrastiveLoss, HuberLoss, LogCoshLoss, XTanhLoss, XSigmoidLoss, QuantileLoss, PearsonCorrelationLoss
from ..modules.transformer import BaseClfHead, BERTClfHead, GPTClfHead, TransformXLClfHead
from ..modules.varclf import OntoBERTClfHead


class Configurable(object):
    # Task related parameters
    TASK_CONFIG_TEMPLATE_KEYS = ['task_type', 'task_ds', 'task_col', 'task_trsfm', 'task_ext_trsfm', 'task_ext_params']
    TASK_CONFIG_TEMPLATE_VALUES = {
        'nmt-base':['nmt', NERDataset, {'index':'id', 'X':'text', 'y':'label'}, (['_nmt_transform'], [{}]), ([_padtrim_transform], [{}]), {'ypad_val':'O', 'trimlbs':True, 'mdlcfg':{'maxlen':128}, 'ignored_label':'O'}],
        'mltc-base': ['mltc-clf', BaseDataset, {'index':'id', 'X':'text', 'y':'label'}, (['_mltc_transform'], [{}]), ([_trim_transform, _sentclf_transform, _pad_transform], [{},{},{}]), {'mdlcfg':{'maxlen':128}, 'ignored_label':'false'}],
        'entlmnt-base': ['entlmnt', EntlmntDataset, {'index':'id', 'X':['sentence1','sentence2'], 'y':'label'}, (['_mltc_transform'], [{}]), ([_trim_transform, _entlmnt_transform, _pad_transform], [{},{},{}]), {'mdlcfg':{'sentsim_func':None, 'concat_strategy':None, 'maxlen':128}}],
        'sentsim-base': ['sentsim', SentSimDataset, {'index':'id', 'X':['sentence1','sentence2'], 'y':'score'}, ([], []), ([_trim_transform, _sentsim_transform, _pad_transform], [{},{},{}]), {'binlb':'rgrsn', 'mdlcfg':{'sentsim_func':None, 'ymode':'sim', 'loss':'contrastive', 'maxlen':128}, 'ynormfunc':(lambda x: x, lambda x: x)}]
    }
    TASK_CONFIG_TEMPLATE_DEFAULTS = TASK_CONFIG_TEMPLATE_VALUES['mltc-base']
    PREDEFINED_TASK_CONFIG_KEYS = ['template', 'task_path', 'kwargs']
    PREDEFINED_TASK_CONFIG_VALUES = {
        'bc5cdr-chem': ['blue-ner', 'BC5CDR-chem', {}],
        'bc5cdr-dz': ['blue-ner', 'BC5CDR-disease', {}],
        'shareclefe': ['blue-ner', 'ShAReCLEFEHealthCorpus', {}],
        'copdner': ['nmt-base', 'copdner', {}],
        'copdnen': ['nmt-base', 'copdner', {'task_trsfm':(['_mltl_nmt_transform'], [{'get_lb': lambda x: '' if x is np.nan or x is None else x.split(';')[0]}])}],
        'ddi': ['blue-mltc', 'ddi2013-type', {'task_ext_params':{'ignored_label':'DDI-false'}}],
        'chemprot': ['blue-mltc', 'ChemProt', {}],
        'i2b2': ['blue-mltc', 'i2b2-2010', {}],
        'hoc': ['blue-mltl', 'hoc', {'task_col':{'index':'index', 'X':'sentence', 'y':'labels'}, 'task_trsfm':(['_mltl_transform'], [{ 'get_lb':lambda x: [s.split('_')[0] for s in x.split(',') if s.split('_')[1] == '1'], 'binlb': dict([(str(x),x) for x in range(10)])}]), 'task_ext_params':{'binlb': OrderedDict([(str(x),x) for x in range(10)]), 'mdlcfg':{'maxlen':128}}}],
        'biolarkgsc': ['mltl-base', 'biolarkgsc', {}],
        'copd': ['mltl-base', 'copd', {}],
        'meshpubs': ['mltl-base', 'meshpubs', {}],
        'phenochf': ['mltl-base', 'phenochf', {}],
        'toxic': ['mltl-base', 'toxic', {}],
        'mednli': ['entlmnt-base', 'mednli', {}],
        'snli': ['glue-entlmnt', 'snli', {}],
        'mnli': ['glue-entlmnt', 'mnli', {}],
        'wnli': ['glue-entlmnt', 'wnli', {'task_col':{'index':'index', 'X':['sentence1','sentence2'], 'y':'label'}}],
        'qnli': ['glue-entlmnt', 'qnli', {'task_col':{'index':'index', 'X':['question','sentence'], 'y':'label'}}],
        'biosses': ['blue-sentsim', 'BIOSSES', {}],
        'clnclsts': ['blue-sentsim', 'clinicalSTS', {}],
    }
    # Model related parameters
    MODEL_CONFIG_TEMPLATE_KEYS = ['encode_func', 'clf_ext_params', 'optmzr', 'shell']
    MODEL_CONFIG_TEMPLATE_VALUES = {
        'transformer-base': [_base_encode, {'lm_loss':False, 'fchdim':0, 'iactvtn':'relu', 'oactvtn':'sigmoid', 'pdrop':0.2, 'do_norm':True, 'norm_type':'batch', 'do_extlin':False, 'do_lastdrop':True, 'do_crf':False, 'initln':False, 'initln_mean':0., 'initln_std':0.02, 'sample_weights':False}, (AdamW, {'correct_bias':False}, 'linwarm'), BaseShell],
        'nn-base': [_tokenize, {'pooler':False, 'seq2seq':'isa', 'seq2vec':'boe', 'fchdim':768, 'iactvtn':'relu', 'oactvtn':'sigmoid', 'w2v_path':None, 'pdrop':0.2, 'do_norm':True, 'norm_type':'batch', 'do_lastdrop':True, 'do_crf':False, 'initln':False, 'initln_mean':0., 'initln_std':0.02, 'sample_weights':False}, None, BaseShell]
    }
    MODEL_CONFIG_TEMPLATE_DEFAULTS = MODEL_CONFIG_TEMPLATE_VALUES['transformer-base']
    PREDEFINED_MODEL_CONFIG_KEYS = ['template', 'lm_mdl_name', 'lm_model', 'lm_params', 'clf', 'config', 'tknzr', 'lm_tknz_extra_char', 'kwargs']
    PREDEFINED_MODEL_CONFIG_VALUES = {
        'bert': ['transformer-base', 'bert-base-uncased', BertModel, 'BERT', BERTClfHead, BertConfig, BertTokenizer, ['[CLS]', '[SEP]', '[SEP]', '[MASK]'], {}],
        'gpt': ['transformer-base', 'openai-gpt', OpenAIGPTLMHeadModel, 'GPT', GPTClfHead, OpenAIGPTConfig, OpenAIGPTTokenizer, ['_@_', ' _$_', ' _#_'], {}],
        'gpt2': ['transformer-base', 'gpt2', GPT2LMHeadModel, 'GPT-2', GPTClfHead, GPT2Config, GPT2Tokenizer, ['_@_', ' _$_', ' _#_'], {}],
        'trsfmxl': ['transformer-base', 'transfo-xl-wt103', TransfoXLLMHeadModel, 'TransformXL', TransformXLClfHead, TransfoXLConfig, TransfoXLTokenizer, ['_@_', ' _$_', ' _#_'], {}],
        'elmo': ['nn-base', 'elmo', Elmo, 'ELMo', {'pool':EmbeddingPool, 's2v':EmbeddingSeq2Vec, 's2s':EmbeddingSeq2Seq, 'ss2v':SentVecEmbeddingSeq2Vec}, elmo_config, None, None, {'embed_type':'elmo'}]
    }
    # Common parameters
    TEMPLATE_VALUES_TYPE_MAP = {'task':TASK_CONFIG_TEMPLATE_VALUES, 'model':MODEL_CONFIG_TEMPLATE_VALUES}

    def __init__(self, dataset, model, template='', **kwargs):
        self.dataset = dataset
        self.model = model
        # Init some templates that based on the existing ones
        Configurable.TASK_CONFIG_TEMPLATE_VALUES['blue-ner'] = Configurable._update_template_values('task', 'nmt-base', {'task_col':{'index':False, 'X':'0', 'y':'3'}})
        Configurable.TASK_CONFIG_TEMPLATE_VALUES['blue-mltc'] = Configurable._update_template_values('task', 'mltc-base', {'task_col':{'index':'index', 'X':'sentence', 'y':'label'}})
        Configurable.TASK_CONFIG_TEMPLATE_VALUES['mltl-base'] = Configurable._update_template_values('task', 'mltc-base', {'task_col':{'index':'id', 'X':'text', 'y':'labels'}, 'task_type':'mltl-clf', 'task_trsfm':(['_mltl_transform'], [{'get_lb':lambda x: [] if x is np.nan or x is None else x.split(kwargs.setdefault('sc', ';;'))}]), 'task_ext_params':{'binlb': 'mltl%s;'%kwargs.setdefault('sc', ';;'), 'mdlcfg':{'maxlen':128}, 'mltl':True}})
        Configurable.TASK_CONFIG_TEMPLATE_VALUES['blue-mltl'] = Configurable._update_template_values('task', 'mltl-base', {'task_col':{'index':'index', 'X':'sentence', 'y':'label'}})
        Configurable.TASK_CONFIG_TEMPLATE_VALUES['glue-entlmnt'] = Configurable._update_template_values('task', 'entlmnt-base', {'task_col':{'index':'index', 'X':['sentence1','sentence2'], 'y':'gold_label'}})
        Configurable.TASK_CONFIG_TEMPLATE_VALUES['blue-sentsim'] = Configurable._update_template_values('task', 'sentsim-base', {'task_col':{'index':'index', 'X':['sentence1','sentence2'], 'y':'score'}, 'task_ext_params':{'binlb':'rgrsn', 'mdlcfg':{'sentsim_func':None, 'ymode':'sim', 'loss':'contrastive', 'maxlen':128}, 'ynormfunc':(lambda x: x / 5.0, lambda x: 5.0 * x)}})
        # Configurable.MODEL_CONFIG_TEMPLATE_VALUES['new_template_name'] = Configurable._update_template_values('model', 'base_model_template_name', {})
        # Instantiation the parameters from template to the properties
        if self.dataset in Configurable.PREDEFINED_TASK_CONFIG_VALUES:
            task_template = Configurable.PREDEFINED_TASK_CONFIG_VALUES[dataset][0]
        if task_template in Configurable.TASK_CONFIG_TEMPLATE_VALUES:
            self.__dict__.update(dict(zip(Configurable.TASK_CONFIG_TEMPLATE_KEYS, Configurable.TASK_CONFIG_TEMPLATE_VALUES[task_template])))
        if self.model in Configurable.PREDEFINED_MODEL_CONFIG_VALUES:
            model_template = Configurable.PREDEFINED_MODEL_CONFIG_VALUES[model][0]
        if model_template in Configurable.MODEL_CONFIG_TEMPLATE_VALUES:
            self.__dict__.update(dict(zip(Configurable.MODEL_CONFIG_TEMPLATE_KEYS, Configurable.MODEL_CONFIG_TEMPLATE_VALUES[model_template])))
        # Config some non-template attributes or overcome the replace the template values from the predefined parameters
        if self.dataset in Configurable.PREDEFINED_TASK_CONFIG_VALUES:
            self.__dict__.update(dict(zip(Configurable.PREDEFINED_TASK_CONFIG_KEYS[1:-1], Configurable.PREDEFINED_TASK_CONFIG_VALUES[self.dataset][1:-1])))
            self.__dict__.update(Configurable.PREDEFINED_TASK_CONFIG_VALUES[self.dataset][-1])
        if self.model in Configurable.PREDEFINED_MODEL_CONFIG_VALUES:
            self.__dict__.update(dict(zip(Configurable.PREDEFINED_MODEL_CONFIG_KEYS[1:-1], Configurable.PREDEFINED_MODEL_CONFIG_VALUES[self.model][1:-1])))
            self.__dict__.update(Configurable.PREDEFINED_MODEL_CONFIG_VALUES[self.model][-1])

        for k, v in kwargs.items():
            setattr(self, k, v)

    def _update_template_values(type, template_name, values={}):
        template_values = copy.deepcopy(Configurable.TEMPLATE_VALUES_TYPE_MAP[type][template_name])
        key_map = dict(zip(Configurable.TASK_CONFIG_TEMPLATE_KEYS, range(len(Configurable.TASK_CONFIG_TEMPLATE_KEYS))))
        for k, v in values.items():
            template_values[key_map[k]] = v
        return template_values

    def _get_template_value(self, name, idx):
        try:
        	return Configurable.TASK_CONFIG_TEMPLATE_VALUES[Configurable.PREDEFINED_TASK_CONFIG_VALUES[self.dataset][0]][idx] if self.dataset in Configurable.PREDEFINED_TASK_CONFIG_VALUES else self.__dict__.setdefault(name, Configurable.TASK_CONFIG_TEMPLATE_DEFAULTS[idx])
        except ValueError as e:
        	return Configurable.MODEL_CONFIG_TEMPLATE_VALUES[Configurable.PREDEFINED_MODEL_CONFIG_VALUES[self.model][0]][idx] if self.model in Configurable.PREDEFINED_MODEL_CONFIG_VALUES else self.__dict__.setdefault(name, Configurable.MODEL_CONFIG_TEMPLATE_DEFAULTS[idx])


    def __getattr__(self, name):
        if name in self.__dict__: return self.__dict__[name]
        try:
            try:
                attr_idx = Configurable.TASK_CONFIG_TEMPLATE_KEYS.index(name)
            except ValueError as e:
                attr_idx = Configurable.MODEL_CONFIG_TEMPLATE_KEYS.index(name)
            return self.__dict__.setdefault(name, self._get_template_value(name, attr_idx))
        except ValueError as e:
            return self.__dict__.setdefault(name, self.__dict__['_'+name] if '_'+name in self.__dict__ else None)


### Module Class and Parameter Config ###
def elmo_config(options_path, weights_path, elmoedim=1024, dropout=0.5):
    return {'options_file':options_path, 'weight_file':weights_path, 'num_output_representations':2, 'elmoedim':elmoedim, 'dropout':dropout}


PYTORCH_WRAPPER = {'lstm':nn.LSTM, 'rnn':nn.RNN, 'gru':nn.GRU, 'agmnlstm':AugmentedLstm, 'stkaltlstm':StackedAlternatingLstm}
SEQ2SEQ_MAP = {'ff':FeedForwardEncoder, 'pytorch':PytorchSeq2SeqWrapper, 'cnn':GatedCnnEncoder, 'isa':IntraSentenceAttentionEncoder, 'qanet':QaNetEncoder, 'ssae':StackedSelfAttentionEncoder}
SEQ2SEQ_MDL_PARAMS = {'pytorch':{'elmo':{'lstm':{'input_size':2048,'hidden_size':768, 'batch_first':True}, 'rnn':{'input_size':2048,'hidden_size':768, 'batch_first':True}, 'gru':{'input_size':2048,'hidden_size':768, 'batch_first':True},'agmnlstm':{'input_size':2048,'hidden_size':768},'stkaltlstm':{'input_size':2048,'hidden_size':768, 'num_layers':3}}}, 'cnn':{'elmo':{'input_dim':2048, 'dropout':0.5, 'layers':[[[4, 2048]],[[4, 2048],[4, 2048]]]}}, 'isa':{'elmo':{'input_dim':2048}}, 'qanet':{'elmo':{}}, 'ssae':{'elmo':{'input_dim':2048, 'hidden_dim':1024, 'projection_dim':768, 'feedforward_hidden_dim':768, 'num_layers':1, 'num_attention_heads':8}}}
SEQ2SEQ_TASK_PARAMS = {}
SEQ2VEC_MAP = {'boe':BagOfEmbeddingsEncoder, 'pytorch':PytorchSeq2VecWrapper, 'allennlp':Seq2VecEncoder, 'cnn':CnnEncoder, 'cnn_highway':CnnHighwayEncoder}
SEQ2VEC_MDL_PARAMS = { \
	'boe':{ \
		'w2v':{'embedding_dim':768, 'averaged':True}, \
		'elmo':{'embedding_dim':768, 'averaged':True}, \
        'elmo_w2v':{'embedding_dim':768, 'averaged':True} \
	}, \
	'pytorch':{ \
		'w2v':{ \
			'lstm':{'input_size':100,'hidden_size':768, 'batch_first':True}, \
			'rnn':{'input_size':100,'hidden_size':768, 'batch_first':True}, \
			'gru':{'input_size':100,'hidden_size':768, 'batch_first':True}, \
			'agmnlstm':{'input_size':100,'hidden_size':768}, \
			'stkaltlstm':{'input_size':100,'hidden_size':768, 'num_layers':3} \
		}, \
		'elmo':{ \
			'lstm':{'input_size':2048,'hidden_size':768, 'batch_first':True}, \
			'rnn':{'input_size':2048,'hidden_size':768, 'batch_first':True}, \
			'gru':{'input_size':2048,'hidden_size':768, 'batch_first':True}, \
			'agmnlstm':{'input_size':2048,'hidden_size':768}, \
			'stkaltlstm':{'input_size':2048,'hidden_size':768, 'num_layers':3} \
		}, \
        'elmo_w2v':{ \
			'lstm':{'input_size':2048,'hidden_size':768, 'batch_first':True}, \
			'rnn':{'input_size':2048,'hidden_size':768, 'batch_first':True}, \
			'gru':{'input_size':2048,'hidden_size':768, 'batch_first':True}, \
			'agmnlstm':{'input_size':2048,'hidden_size':768}, \
			'stkaltlstm':{'input_size':2048,'hidden_size':768, 'num_layers':3} \
		} \
	}, \
	'cnn':{ \
		'w2v':{'embedding_dim':100, 'num_filters':768}, \
		'elmo':{'embedding_dim':2048, 'num_filters':768}, \
        'elmo_w2v':{'embedding_dim':2048, 'num_filters':768} \
	}, \
	'cnn_highway':{ \
		'w2v':{'embedding_dim':100, 'filters':[(2, 768),(3, 768),(4, 768),(5, 768)], 'num_highway':5, 'projection_dim':100}, \
		'elmo':{'embedding_dim':2048, 'filters':[(2, 768),(3, 768),(4, 768),(5, 768)], 'num_highway':5, 'projection_dim':2048}, \
        'elmo_w2v':{'embedding_dim':2048, 'filters':[(2, 768),(3, 768),(4, 768),(5, 768)], 'num_highway':5, 'projection_dim':2048} \
	} \
}
SEQ2VEC_TASK_PARAMS = {}
SEQ2VEC_LM_PARAMS_MAP = {'boe':[('hdim','embedding_dim')], 'pytorch':[('hdim', 'hidden_size')], 'cnn':[], 'cnn_highway':[]}
SEQ2SEQ_DIM_INFER = {'pytorch-lstm':lambda x: x[1] * x[2]['hidden_size'], 'pytorch-rnn':lambda x: x[1] * x[2]['hidden_size'], 'pytorch-gru':lambda x: x[1] * x[2]['hidden_size'], 'cnn':lambda x: 2 * x[0], 'isa':lambda x: x[0]}
SEQ2VEC_DIM_INFER = {'boe':lambda x: x[0], 'pytorch-lstm':lambda x: x[2]['hidden_size'], 'pytorch-agmnlstm':lambda x: x[2]['hidden_size'], 'pytorch-rnn':lambda x: x[2]['hidden_size'], 'pytorch-stkaltlstm':lambda x: x[2]['hidden_size'], 'pytorch-gru':lambda x: x[2]['hidden_size'], 'cnn':lambda x: int(1.5 * x[2]['embedding_dim']), 'cnn_highway':lambda x: x[0]}
NORM_TYPE_MAP = {'batch':nn.BatchNorm1d, 'layer':nn.LayerNorm}
ACTVTN_MAP = {'relu':nn.ReLU, 'sigmoid':nn.Sigmoid, 'tanh':nn.Tanh}
RGRSN_LOSS_MAP = {'mse':MSELoss, 'contrastive':ContrastiveLoss, 'huber':HuberLoss, 'logcosh':LogCoshLoss, 'xtanh':XTanhLoss, 'xsigmoid':XSigmoidLoss, 'quantile':QuantileLoss, 'pearson':PearsonCorrelationLoss}
SIM_FUNC_MAP = {'sim':'sim', 'dist':'dist'}
CNSTRNT_PARAMS_MAP = {'hrch':'Hrch'}
CNSTRNTS_MAP = {'hrch':(HrchConstraint, {('num_lbs','num_lbs'):1, ('hrchrel_path','hrchrel_path'):'hpo_ancrels.pkl', ('binlb','binlb'):{}})}



### Unit Test ###
def test_config():
    config = Configurable('biolarkgsc', 'bert')
    print(config.__dict__)