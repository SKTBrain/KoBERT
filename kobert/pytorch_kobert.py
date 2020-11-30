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

import os
from zipfile import ZipFile

import torch

from transformers import BertModel
import gluonnlp as nlp

from .utils import download as _download
from .utils import tokenizer

pytorch_kobert = {
    'url':
    'https://kobert.blob.core.windows.net/models/kobert/pytorch/kobert_v1.zip',
    'fname': 'kobert_v1.zip',
    'chksum': '411b242919'  # 411b2429199bc04558576acdcac6d498
}


def get_pytorch_kobert_model(ctx='cpu', cachedir='~/kobert/'):
    # download model
    model_info = pytorch_kobert
    model_down = _download(model_info['url'],
                           model_info['fname'],
                           model_info['chksum'],
                           cachedir=cachedir)
    
    zipf = ZipFile(os.path.expanduser(model_down))
    zipf.extractall(path=os.path.expanduser(cachedir))
    model_path = os.path.join(os.path.expanduser(cachedir), 'kobert_from_pretrained')
    # download vocab
    vocab_info = tokenizer
    vocab_path = _download(vocab_info['url'],
                           vocab_info['fname'],
                           vocab_info['chksum'],
                           cachedir=cachedir)
    return get_kobert_model(model_path, vocab_path, ctx)


def get_kobert_model(model_path, vocab_file, ctx="cpu"):
    bertmodel = BertModel.from_pretrained(model_path)
    device = torch.device(ctx)
    bertmodel.to(device)
    bertmodel.eval()
    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                         padding_token='[PAD]')
    return bertmodel, vocab_b_obj
