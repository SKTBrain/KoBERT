# coding=utf-8
# Copyright 2019 SK T-Brain Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gluonnlp as nlp
import mxnet as mx
from gluonnlp.model import BERTEncoder, BERTModel

from .utils import download as _download
from .utils import tokenizer

mxnet_kobert = {
    'url':
    'https://kobert.blob.core.windows.net/models/kobert/mxnet/mxnet_kobert_45b6957552.params',
    'fname': 'mxnet_kobert_45b6957552.params',
    'chksum': '45b6957552'
}


def get_mxnet_kobert_model(use_pooler=True,
                           use_decoder=True,
                           use_classifier=True,
                           ctx=mx.cpu(0),
                           cachedir='~/kobert/'):
    # download model
    model_info = mxnet_kobert
    model_path = _download(model_info['url'],
                           model_info['fname'],
                           model_info['chksum'],
                           cachedir=cachedir)
    # download vocab
    vocab_info = tokenizer
    vocab_path = _download(vocab_info['url'],
                           vocab_info['fname'],
                           vocab_info['chksum'],
                           cachedir=cachedir)
    return get_kobert_model(model_path, vocab_path, use_pooler, use_decoder,
                            use_classifier, ctx)


def get_kobert_model(model_file,
                     vocab_file,
                     use_pooler=True,
                     use_decoder=True,
                     use_classifier=True,
                     ctx=mx.cpu(0)):
    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                         padding_token='[PAD]')

    predefined_args = {
        'attention_cell': 'multi_head',
        'num_layers': 12,
        'units': 768,
        'hidden_size': 3072,
        'max_length': 512,
        'num_heads': 12,
        'scaled': True,
        'dropout': 0.1,
        'use_residual': True,
        'embed_size': 768,
        'embed_dropout': 0.1,
        'token_type_vocab_size': 2,
        'word_embed': None,
    }

    encoder = BERTEncoder(num_layers=predefined_args['num_layers'],
                          units=predefined_args['units'],
                          hidden_size=predefined_args['hidden_size'],
                          max_length=predefined_args['max_length'],
                          num_heads=predefined_args['num_heads'],
                          dropout=predefined_args['dropout'],
                          output_attention=False,
                          output_all_encodings=False)

    # BERT
    net = BERTModel(
        encoder,
        len(vocab_b_obj.idx_to_token),
        token_type_vocab_size=predefined_args['token_type_vocab_size'],
        units=predefined_args['units'],
        embed_size=predefined_args['embed_size'],
        word_embed=predefined_args['word_embed'],
        use_pooler=use_pooler,
        use_decoder=use_decoder,
        use_classifier=use_classifier)
    net.initialize(ctx=ctx)
    net.load_parameters(model_file, ctx, ignore_extra=True)
    return (net, vocab_b_obj)
