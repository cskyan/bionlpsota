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

import copy, logging
from collections import OrderedDict

import numpy as np

import torch
from torch import nn

from allennlp.modules.elmo import Elmo
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.stacked_alternating_lstm import StackedAlternatingLstm
from allennlp.modules.seq2seq_encoders import FeedForwardEncoder, PytorchSeq2SeqWrapper, GatedCnnEncoder, IntraSentenceAttentionEncoder, QaNetEncoder, StackedSelfAttentionEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, PytorchSeq2VecWrapper, Seq2VecEncoder, CnnEncoder, CnnHighwayEncoder

from transformers import BertConfig, BertTokenizer, BertModel, AdamW, OpenAIGPTConfig, OpenAIGPTTokenizer, OpenAIGPTModel, GPT2Config, GPT2Tokenizer, GPT2Model, TransfoXLConfig, TransfoXLTokenizer, TransfoXLModel

from modules.shell import BaseShell
from modules.constraint import HrchConstraint
from modules.embedding import EmbeddingPool, EmbeddingSeq2Vec, EmbeddingSeq2Seq, SentVecEmbeddingSeq2Vec
from modules.loss import MSELoss, ContrastiveLoss, HuberLoss, LogCoshLoss, XTanhLoss, XSigmoidLoss, QuantileLoss, PearsonCorrelationLoss
from modules.transformer import BERTClfHead, GPTClfHead, TransformXLClfHead
from .dataset import BaseDataset, SentSimDataset, EntlmntDataset, NERDataset
from . import processor as P


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


### Universal Config Class ###
class Configurable(object):
    # Task related parameters
    TASK_CONFIG_TEMPLATE_KEYS = ['task_type', 'task_ds', 'task_col', 'task_trsfm', 'task_ext_params', 'ds_kwargs']
    TASK_CONFIG_TEMPLATE_VALUES = {
        'nmt-base':['nmt', NERDataset, {'index':'id', 'X':'text', 'y':'label'}, (['_nmt_transform'], [{}]), {'ypad_val':'O', 'trimlbs':True, 'mdlaware':{}, 'ignored_label':'O'}, {}],
        'mltc-base': ['mltc-clf', BaseDataset, {'index':'id', 'X':'text', 'y':'label'}, (['_mltc_transform'], [{}]), {'mdlaware':{}, 'ignored_label':'false'}, {}],
        'entlmnt-base': ['entlmnt', EntlmntDataset, {'index':'id', 'X':['sentence1','sentence2'], 'y':'label'}, (['_mltc_transform'], [{}]), {'mdlaware':{'sentsim_func':None, 'concat_strategy':None}}, {}],
        'sentsim-base': ['sentsim', SentSimDataset, {'index':'id', 'X':['sentence1','sentence2'], 'y':'score'}, ([], []), {'binlb':'rgrsn', 'mdlaware':{'sentsim_func':None, 'ymode':'sim', 'loss':'contrastive'}, 'ynormfunc':(lambda x: x, lambda x: x)}, {}]
    }
    TASK_CONFIG_TEMPLATE_DEFAULTS = TASK_CONFIG_TEMPLATE_VALUES['mltc-base']
    TASK_CONFIG_TEMPLATE_EXPRESSION = {
    }
    TASK_CONFIG_TEMPLATE_UPDATES = {
        'all': [
            ('ds_kwargs', lambda param, val: param.update({'sampw':val['sample_weights']}), {'sample_weights':'property'}),
            ('ds_kwargs', lambda param, val: param.update({'sampfrac':val['sampfrac']}), {'sampfrac':'property'})
        ],
        'nmt-base': [
            ('ds_kwargs', lambda param, val: param.update({'lb_coding':val['task_ext_params'].setdefault('lb_coding', 'IOB')}), {'task_ext_params':'property'})
        ],
        'entlmnt-base': [
            ('ds_kwargs', lambda param, val: param.update(dict((k, val['task_ext_params'][k]) for k in ['origlb', 'lbtxt', 'neglbs', 'reflb'] if k in val)), {'task_ext_params':'property'})
        ],
        'sentsim-base': [
            ('ds_kwargs', lambda param, val: param.update({'ynormfunc':val['task_ext_params'].setdefault('ynormfunc', None)}), {'task_ext_params':'property'})
        ]
    }
    TASK_CONFIG_TEMPLATE_DELAYED_UPDATES = {
    }
    TASK_CONFIG_EXHAUSTED_UPDATES = {
        'all': [
            ('task_ext_params', lambda param, val: (param.setdefault('mdlaware', {}), val))
        ]
    }
    PREDEFINED_TASK_CONFIG_KEYS = ['template', 'task_path', 'kwargs']
    PREDEFINED_TASK_CONFIG_VALUES = {
        'bc5cdr-chem': ['blue-ner', 'BC5CDR-chem', {}],
        'bc5cdr-dz': ['blue-ner', 'BC5CDR-disease', {}],
        'shareclefe': ['blue-ner', 'ShAReCLEFEHealthCorpus', {}],
        'copdner': ['nmt-base', 'copdner', {}],
        'copdnen': ['nmt-base', 'copdner', {'task_trsfm':(['_mltl_nmt_transform'], [{'get_lb': lambda x: '' if x is np.nan or x is None else x.split(';')[0]}])}],
        'ddi': ['blue-mltc', 'ddi2013-type', {}],
        'chemprot': ['blue-mltc', 'ChemProt', {}],
        'i2b2': ['blue-mltc', 'i2b2-2010', {}],
        'hoc': ['blue-mltl', 'hoc', {'task_col':{'index':'index', 'X':'sentence', 'y':'labels'}, 'task_trsfm':(['_mltl_transform'], [{'get_lb':lambda x: [s.split('_')[0] for s in x.split(',') if s.split('_')[1] == '1']}]), 'task_ext_params':{'binlb': OrderedDict([(str(x),x) for x in range(10)]), 'mdlaware':{}}}],
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
        'hpo_entilement': ['entlmnt-base', 'hpo.entlmnt', {'task_col':{'index':'id', 'X':['text1', 'onto'], 'y':'label', 'ontoid':'label2'}}],
        'biolarkgsc_entilement': ['entlmnt-base', 'biolarkgsc.entlmnt', {'task_col':{'index':'id', 'X':['text1', 'onto'], 'y':'label', 'ontoid':'label2'}}]
    }
    PREDEFINED_TASK_CONFIG_EXPRESSION = {
    }
    PREDEFINED_TASK_CONFIG_UPDATES = {
        'ddi': [
            ('task_ext_params', lambda param, val: param.update({'ignored_label':'DDI-false'}), {})
        ],
        'hpo_entilement': [
            ('ds_kwargs', lambda param, val: param.update({'onto_fpath':val['onto'] if val['onto'] and os.path.exists(val['onto']) else 'onto.csv'}), {'onto':'property'}),
            ('ds_kwargs', lambda param, val: param.update({'onto_col':val['ontoid']}), {'task_col':'property'}),
            ('tknz_kwargs', lambda param, val: param.update({'truncation':'only_first'}), {})
        ],
        'biolarkgsc_entilement': [
            ('ds_kwargs', lambda param, val: param.update({'onto_fpath':val['onto'] if val['onto'] and os.path.exists(val['onto']) else 'onto.csv'}), {'onto':'property'}),
            ('ds_kwargs', lambda param, val: param.update({'onto_col':val['ontoid']}), {'task_col':'property'})
        ]
    }
    PREDEFINED_TASK_CONFIG_DELAYED_UPDATES = {
    }
    # Model related parameters
    MODEL_CONFIG_TEMPLATE_KEYS = ['encode_func', 'tknz_kwargs', 'mdl_trsfm', 'clf_ext_params', 'optmzr', 'shell']
    MODEL_CONFIG_TEMPLATE_VALUES = {
        'transformer-base': [P._base_encode, {'max_length':128, 'padding':'max_length', 'truncation': True}, ([P._base_transform],[{'input_keys':['input_ids']}]), {'lm_loss':False, 'fchdim':0, 'extfc':False, 'iactvtn':'relu', 'oactvtn':'sigmoid', 'do_drop':False, 'pdrop':0.2, 'do_norm':True, 'norm_type':'batch', 'do_extlin':False, 'do_lastdrop':True, 'do_crf':False, 'initln':False, 'initln_mean':0., 'initln_std':0.02, 'sample_weights':False}, (AdamW, {'correct_bias':False}, 'linwarm'), BaseShell],
        'nn-base': [P._tokenize, {}, ([P._nn_transform],[{}]), {'pooler':False, 'seq2seq':'isa', 'seq2vec':'boe', 'fchdim':768, 'iactvtn':'relu', 'oactvtn':'sigmoid', 'w2v_path':None, 'pdrop':0.2, 'do_norm':True, 'norm_type':'batch', 'do_lastdrop':True, 'do_crf':False, 'initln':False, 'initln_mean':0., 'initln_std':0.02, 'sample_weights':False}, None, BaseShell]
    }
    MODEL_CONFIG_TEMPLATE_DEFAULTS = MODEL_CONFIG_TEMPLATE_VALUES['transformer-base']
    MODEL_CONFIG_TEMPLATE_EXPRESSION = {
        'transformer-base': [('input_keys', lambda x: ['input_ids', 'attention_mask'], {'x':None})],
        'nn-base': [('input_keys', lambda x: ['input_ids', 'mask'], {'x':None})],
    }
    MODEL_CONFIG_TEMPLATE_UPDATES = {
        'transformer-base': [
            ('tknz_kwargs', lambda param, val: param.update({'max_length':val['maxlen']}), {'maxlen':'property'}),
            ('mdl_trsfm', lambda param, val: param[1][0].update({'input_keys':val['input_keys']}), {'input_keys':'property'})
        ],
        'nn-base': [
            ('mdl_trsfm', lambda param, val: param[1][0].update({'maxlen':val['maxlen'], 'pad_token':val['tknzr'].setdefault('pad_token', '<PAD>') if type(val['tknzr']) is dict else (val['tknzr'].pad_token if hasattr(val['tknzr'], 'pad_token') else '<PAD>')}), {'maxlen':'property', 'tknzr':'property'}),
            ('clf_ext_params', lambda param, val: param.update({'embed_type':val['embed_type']}), {'embed_type':'property-skipna'})
        ]
    }
    MODEL_CONFIG_TEMPLATE_DELAYED_UPDATES = {
    }
    MODEL_CONFIG_EXHAUSTED_UPDATES = {
        'all': [
            ('clf_ext_params', lambda param, val: (param, val))
        ]
    }
    PREDEFINED_MODEL_CONFIG_KEYS = ['template', 'lm_mdl_name', 'lm_model', 'lm_config', 'lm_params', 'clf', 'tknzr', 'kwargs']
    PREDEFINED_MODEL_CONFIG_VALUES = {
        'bert': ['transformer-base', 'bert-base-uncased', BertModel, BertConfig, 'BERT', BERTClfHead, BertTokenizer, {}],
        'gpt': ['transformer-base', 'openai-gpt', OpenAIGPTModel, OpenAIGPTConfig, 'GPT', GPTClfHead, OpenAIGPTTokenizer, {}],
        'gpt2': ['transformer-base', 'gpt2', GPT2Model, GPT2Config, 'GPT-2', GPTClfHead, GPT2Tokenizer, {}],
        'trsfmxl': ['transformer-base', 'transfo-xl-wt103', TransfoXLModel, TransfoXLConfig, 'TransformXL', TransformXLClfHead, TransfoXLTokenizer, {}],
        # Transformer-based approach should not have embed_type attribute
        'elmo': ['nn-base', 'elmo', Elmo, elmo_config, 'ELMo', {'pool':EmbeddingPool, 's2v':EmbeddingSeq2Vec, 's2s':EmbeddingSeq2Seq, 'ss2v':SentVecEmbeddingSeq2Vec}, None, {'embed_type':'elmo'}],
        'none': ['nn-base', 'embedding', None, None, 'Embedding', {'pool':EmbeddingPool, 's2v':EmbeddingSeq2Vec, 's2s':EmbeddingSeq2Seq, 'ss2v':SentVecEmbeddingSeq2Vec}, None, {'embed_type':'w2v'}], # all the embedding-based approaches without ELMo
    }
    PREDEFINED_MODEL_CONFIG_EXPRESSION = {
        'bert': [('input_keys', P._bert_input_keys, {'task_type':'property'})],
        'elmo': [('encoder', lambda encoder: encoder if encoder is not None and not encoder.isspace() else 'pool', {'encoder':'property'})],
        'none': [('input_keys', P._embed_input_keys, {'embed_type':'property'})]
    }
    def set_elmo_transform(param, val):
        param[0][0]=P._elmo_transform
    def set_embed_transform(param, val):
        param[0][0]=P._embedding_transform
        param[1][0].update({'embed_type':val['embed_type']})
    PREDEFINED_MODEL_CONFIG_UPDATES = {
        'bert': [('clf_ext_params', lambda param, val: param.update({'do_drop':val['do_drop']}), {'do_drop':'property'})],
        'elmo': [('mdl_trsfm', set_elmo_transform, {'lm_config':'property'})],
        'none': [('mdl_trsfm', set_embed_transform, {'embed_type':'property'})]
    }
    PREDEFINED_MODEL_CONFIG_DELAYED_UPDATES = {
        'elmo': [('mdl_trsfm', lambda param, val: param[1][0].update({'lm_config':val['lm_config']}), {'lm_config':'property'})],
        'none': [('mdl_trsfm', lambda param, val: param[1][0].update({'w2v_model':val['w2v_model']}), {'w2v_model':'property'})]
    }
    # Common parameters
    TEMPLATE_VALUES_TYPE_MAP = {'task':TASK_CONFIG_TEMPLATE_VALUES, 'model':MODEL_CONFIG_TEMPLATE_VALUES}

    def __init__(self, dataset, model, template='', **kwargs):
        self.dataset = dataset
        self.model = model
        self._callbacks = {}
        # Init some templates that based on the existing ones
        Configurable.TASK_CONFIG_TEMPLATE_VALUES['blue-ner'] = Configurable._update_template_values('task', 'nmt-base', {'task_col':{'index':False, 'X':'0', 'y':'3'}})
        Configurable.TASK_CONFIG_TEMPLATE_VALUES['blue-mltc'] = Configurable._update_template_values('task', 'mltc-base', {'task_col':{'index':'index', 'X':'sentence', 'y':'label'}})
        Configurable.TASK_CONFIG_TEMPLATE_VALUES['mltl-base'] = Configurable._update_template_values('task', 'mltc-base', {'task_col':{'index':'id', 'X':'text', 'y':'labels'}, 'task_type':'mltl-clf', 'task_trsfm':(['_mltl_transform'], [{'get_lb':lambda x: [] if x is np.nan or x is None else x.split(kwargs.setdefault('sc', ';;'))}]), 'task_ext_params':{'binlb': 'mltl%s;'%kwargs.setdefault('sc', ';;'), 'mdlaware':{}, 'mltl':True}})
        Configurable.TASK_CONFIG_TEMPLATE_VALUES['blue-mltl'] = Configurable._update_template_values('task', 'mltl-base', {'task_col':{'index':'index', 'X':'sentence', 'y':'label'}})
        Configurable.TASK_CONFIG_TEMPLATE_VALUES['glue-entlmnt'] = Configurable._update_template_values('task', 'entlmnt-base', {'task_col':{'index':'index', 'X':['sentence1','sentence2'], 'y':'gold_label'}})
        Configurable.TASK_CONFIG_TEMPLATE_VALUES['blue-sentsim'] = Configurable._update_template_values('task', 'sentsim-base', {'task_col':{'index':'index', 'X':['sentence1','sentence2'], 'y':'score'}, 'task_ext_params':{'binlb':'rgrsn', 'mdlaware':{'sentsim_func':None, 'ymode':'sim', 'loss':'contrastive'}, 'ynormfunc':(lambda x: x / 5.0, lambda x: 5.0 * x)}})
        # Configurable.MODEL_CONFIG_TEMPLATE_VALUES['new_template_name'] = Configurable._update_template_values('model', 'base_model_template_name', {})
        # Instantiation the parameters from template to the properties
        if self.dataset in Configurable.PREDEFINED_TASK_CONFIG_VALUES:
            task_template = Configurable.PREDEFINED_TASK_CONFIG_VALUES[self.dataset][0]
            if task_template in Configurable.TASK_CONFIG_TEMPLATE_VALUES:
                self.__dict__.update(dict(zip(Configurable.TASK_CONFIG_TEMPLATE_KEYS, Configurable.TASK_CONFIG_TEMPLATE_VALUES[task_template])))
        if self.model in Configurable.PREDEFINED_MODEL_CONFIG_VALUES:
            model_template = Configurable.PREDEFINED_MODEL_CONFIG_VALUES[model][0]
            if model_template in Configurable.MODEL_CONFIG_TEMPLATE_VALUES:
                self.__dict__.update(dict(zip(Configurable.MODEL_CONFIG_TEMPLATE_KEYS, Configurable.MODEL_CONFIG_TEMPLATE_VALUES[model_template])))
        # Config some non-template attributes or overwrite the template values with the predefined parameters
        if self.dataset in Configurable.PREDEFINED_TASK_CONFIG_VALUES:
            self.__dict__.update(dict(zip(Configurable.PREDEFINED_TASK_CONFIG_KEYS[1:-1], Configurable.PREDEFINED_TASK_CONFIG_VALUES[self.dataset][1:-1])))
            self.__dict__.update(Configurable.PREDEFINED_TASK_CONFIG_VALUES[self.dataset][-1])
        if self.model in Configurable.PREDEFINED_MODEL_CONFIG_VALUES:
            self.__dict__.update(dict(zip(Configurable.PREDEFINED_MODEL_CONFIG_KEYS[1:-1], Configurable.PREDEFINED_MODEL_CONFIG_VALUES[self.model][1:-1])))
            self.__dict__.update(Configurable.PREDEFINED_MODEL_CONFIG_VALUES[self.model][-1])

        # Config the keyword arguments
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Config simple attributes (assign the whole variable) with the return of functions on run-time parameters
        expression_updates = Configurable.TASK_CONFIG_TEMPLATE_EXPRESSION.setdefault('all', []) + Configurable.MODEL_CONFIG_TEMPLATE_EXPRESSION.setdefault('all', [])
        if 'task_template' in locals() and task_template in Configurable.TASK_CONFIG_TEMPLATE_EXPRESSION: expression_updates.extend(Configurable.TASK_CONFIG_TEMPLATE_EXPRESSION[task_template])
        if self.dataset in Configurable.PREDEFINED_TASK_CONFIG_EXPRESSION: expression_updates.extend(Configurable.PREDEFINED_TASK_CONFIG_EXPRESSION[self.dataset])
        if 'model_template' in locals() and model_template in Configurable.MODEL_CONFIG_TEMPLATE_EXPRESSION: expression_updates.extend(Configurable.MODEL_CONFIG_TEMPLATE_EXPRESSION[model_template])
        if self.model in Configurable.PREDEFINED_MODEL_CONFIG_EXPRESSION: expression_updates.extend(Configurable.PREDEFINED_MODEL_CONFIG_EXPRESSION[self.model])
        for attr, exp_func, params in expression_updates:
            try:
                param_vals = self._get_attributes(params)
                setattr(self, attr, exp_func(**param_vals))
            except Exception as e:
                logging.error(e)
                logging.warning('Failed to configurate EXPRESSION attribute [%s] with function [%s] and parameters [%s]' % (attr, exp_func, params))

        # Config complex attributes (parts of a dict) or overwrite the template values with the return of functions on run-time parameters
        partial_updates = Configurable.TASK_CONFIG_TEMPLATE_UPDATES.setdefault('all', []) + Configurable.MODEL_CONFIG_TEMPLATE_UPDATES.setdefault('all', [])
        if 'task_template' in locals() and task_template in Configurable.TASK_CONFIG_TEMPLATE_UPDATES: partial_updates.extend(Configurable.TASK_CONFIG_TEMPLATE_UPDATES[task_template])
        if self.dataset in Configurable.PREDEFINED_TASK_CONFIG_UPDATES: partial_updates.extend(Configurable.PREDEFINED_TASK_CONFIG_UPDATES[self.dataset])
        if 'model_template' in locals() and model_template in Configurable.MODEL_CONFIG_TEMPLATE_UPDATES: partial_updates.extend(Configurable.MODEL_CONFIG_TEMPLATE_UPDATES[model_template])
        if self.model in Configurable.PREDEFINED_MODEL_CONFIG_UPDATES: partial_updates.extend(Configurable.PREDEFINED_MODEL_CONFIG_UPDATES[self.model])
        for attr, exp_func, params in partial_updates:
            try:
                param_vals = self._get_attributes(params)
                exp_func(getattr(self, attr), param_vals)
            except Exception as e:
                logging.error(e)
                logging.warning('Failed to configurate PARTIAL attribute [%s] with function [%s] and parameters [%s]' % (attr, exp_func, params))

        # Update complex attributes (parts of a dict) with the all the possible run-time parameters
        partail_exhausted_updates = Configurable.TASK_CONFIG_EXHAUSTED_UPDATES.setdefault('all', []) + Configurable.MODEL_CONFIG_EXHAUSTED_UPDATES.setdefault('all', [])
        if 'task_template' in locals() and task_template in Configurable.TASK_CONFIG_EXHAUSTED_UPDATES: partail_exhausted_updates.extend(Configurable.TASK_CONFIG_EXHAUSTED_UPDATES[task_template])
        if 'model_template' in locals() and model_template in Configurable.MODEL_CONFIG_EXHAUSTED_UPDATES: partail_exhausted_updates.extend(Configurable.MODEL_CONFIG_EXHAUSTED_UPDATES[model_template])
        for attr, param_func in partail_exhausted_updates:
            try:
                params, vals = param_func(getattr(self, attr), self)
                params.update(dict([(k, vals[k]) if k in vals and vals[k] is not None else (k, v) for k, v in params.items()]) if type (vals) is dict else dict([(k, getattr(vals, k)) if hasattr(vals, k) and getattr(vals, k) is not None else (k, v) for k, v in params.items()]))
            except Exception as e:
                logging.error(e)
                logging.warning('Failed to configurate EXHAUSTED attribute [%s] with parameter function [%s]' % (attr, param_func))
        if self.verbose: logging.debug(dict([(k, v) for k, v in self.__dict__.items() if not k.startswith('_') and not callable(v)]))


    def _update_template_values(type, template_name, values={}):
        template_values = copy.deepcopy(Configurable.TEMPLATE_VALUES_TYPE_MAP[type][template_name])
        key_map = dict(zip(Configurable.TASK_CONFIG_TEMPLATE_KEYS, range(len(Configurable.TASK_CONFIG_TEMPLATE_KEYS))))
        for k, v in values.items():
            template_values[key_map[k]] = v
        return template_values

    # Config complex attributes (partial updates) at a delayed time
    def delayed_update(self, updates):
        for attr, exp_func, params in updates:
            try:
                param_vals = self._get_attributes(params)
                exp_func(getattr(self, attr), param_vals)
            except Exception as e:
                logging.error(e)
                logging.warning('Failed to configurate PARTIAL attribute [%s] with function [%s] and parameters [%s]' % (attr, exp_func, params))

    # Operations that should be done in a stack order (e.g. property injection). Note: sequential operations should be done directly in the object construction function.
    def register_callback(self, callback_name, callback_func, replace=False):
        # if callback_name in self._callbacks and not replace:
        #     logging.error('Falied to register callback function `%s`' % callback_name)
        #     return False
        # self._callbacks[callback_name] = callback_func
        self._callbacks.setdefault(callback_name, []).append(callback_func)
        return True

    def execute_callback(self, callback_name):
        for func in self._callbacks[callback_name][::-1]:
            func(self)

    def execute_all_callback(self):
        for name, func_stack in self._callbacks.items():
            for func in func_stack[::-1]:
                func(self)

    def _get_attributes(self, params):
        param_vals = {}
        for k, v in params.items():
            if v is not None and v.startswith('property'):
                try:
                    param_vals.update({k:getattr(self, k)})
                except Exception as e:
                    if v != 'property-skipna': logging.warning('Unable to get the attribute `%s` from config object' % k)
            else:
                param_vals.update({k:v})
        return param_vals

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


### Unit Test ###
def test_config():
    config = Configurable('biolarkgsc', 'bert')
    print(config.__dict__)
