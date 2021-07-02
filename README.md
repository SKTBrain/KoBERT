

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Korean BERT pre-trained cased (KoBERT)](#korean-bert-pre-trained-cased-kobert)
  - [Why'?'](#why)
  - [Training Environment](#training-environment)
  - [Requirements](#requirements)
  - [How to install](#how-to-install)
- [How to use](#how-to-use)
  - [Using with PyTorch](#using-with-pytorch)
  - [Using with ONNX](#using-with-onnx)
  - [Using with MXNet-Gluon](#using-with-mxnet-gluon)
  - [Tokenizer](#tokenizer)
- [Subtasks](#subtasks)
  - [Naver Sentiment Analysis](#naver-sentiment-analysis)
  - [KoBERT와 CRF로 만든 한국어 객체명인식기](#kobert와-crf로-만든-한국어-객체명인식기)
- [Version History](#version-history)
- [Contacts](#contacts)
- [License](#license)

<!-- /code_chunk_output -->

---

### Korean BERT pre-trained cased (KoBERT)

#### Why'?'

* 구글 [BERT base multilingual cased](https://github.com/google-research/bert/blob/master/multilingual.md)의 한국어 성능 한계

#### Training Environment

* Architecture

```python
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
```

* 학습셋

| 데이터  |  문장  | 단어 |
|---|---|---|
| 한국어 위키  |  5M |  54M  |

* 학습 환경
  * V100 GPU x 32, Horovod(with InfiniBand)

![2019-04-29 텐서보드 로그](imgs/2019-04-29_TensorBoard.png)

* 사전(Vocabulary)
  * 크기 : 8,002
  * 한글 위키 기반으로 학습한 토크나이저(SentencePiece)
  * Less number of parameters(92M < 110M )

#### Requirements

* Python >= 3.6
* PyTorch >= 1.7.0
* MXNet >= 1.4.0
* gluonnlp >= 0.6.0
* sentencepiece >= 0.1.6
* onnxruntime >= 0.3.0
* transformers >= 3.5.0

#### How to install

```sh
git clone https://github.com/SKTBrain/KoBERT.git
cd KoBERT
pip install -r requirements.txt
pip install .
```

---

### How to use

#### Using with PyTorch

*Huggingface transformers API가 편하신 분은 [여기](kobert_hf)를 참고하세요.*

```python
>>> import torch
>>> from kobert.pytorch_kobert import get_pytorch_kobert_model
>>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
>>> input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
>>> token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
>>> model, vocab  = get_pytorch_kobert_model()
>>> sequence_output, pooled_output = model(input_ids, input_mask, token_type_ids)
>>> pooled_output.shape
torch.Size([2, 768])
>>> vocab
Vocab(size=8002, unk="[UNK]", reserved="['[MASK]', '[SEP]', '[CLS]']")
>>> # Last Encoding Layer
>>> sequence_output[0]
tensor([[-0.2461,  0.2428,  0.2590,  ..., -0.4861, -0.0731,  0.0756],
        [-0.2478,  0.2420,  0.2552,  ..., -0.4877, -0.0727,  0.0754],
        [-0.2472,  0.2420,  0.2561,  ..., -0.4874, -0.0733,  0.0765]],
       grad_fn=<SelectBackward>)
```

`model`은 디폴트로 `eval()`모드로 리턴됨, 따라서 학습 용도로 사용시 `model.train()`명령을 통해 학습 모드로 변경할 필요가 있다.

- Naver Sentiment Analysis Fine-Tuning with pytorch
  - Colab에서 [런타임] - [런타임 유형 변경] - 하드웨어 가속기(GPU) 사용을 권장합니다.
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb)


#### Using with ONNX

```python
>>> import onnxruntime
>>> import numpy as np
>>> from kobert.utils import get_onnx
>>> onnx_path = get_onnx()
>>> sess = onnxruntime.InferenceSession(onnx_path)
>>> input_ids = [[31, 51, 99], [15, 5, 0]]
>>> input_mask = [[1, 1, 1], [1, 1, 0]]
>>> token_type_ids = [[0, 0, 1], [0, 1, 0]]
>>> len_seq = len(input_ids[0])
>>> pred_onnx = sess.run(None, {'input_ids':np.array(input_ids),
>>>                             'token_type_ids':np.array(token_type_ids),
>>>                             'input_mask':np.array(input_mask),
>>>                             'position_ids':np.array(range(len_seq))})
>>> # Last Encoding Layer
>>> pred_onnx[-2][0]
array([[-0.24610452,  0.24282141,  0.25895312, ..., -0.48613444,
        -0.07305173,  0.07560554],
       [-0.24783179,  0.24200465,  0.25520486, ..., -0.4877185 ,
        -0.0727044 ,  0.07536091],
       [-0.24721591,  0.24196623,  0.2560626 , ..., -0.48743123,
        -0.07326943,  0.07650235]], dtype=float32)
```

_ONNX 컨버팅은 [soeque1](https://github.com/soeque1)께서 도움을 주셨습니다._

#### Using with MXNet-Gluon

```python
>>> import mxnet as mx
>>> from kobert.mxnet_kobert import get_mxnet_kobert_model
>>> input_id = mx.nd.array([[31, 51, 99], [15, 5, 0]])
>>> input_mask = mx.nd.array([[1, 1, 1], [1, 1, 0]])
>>> token_type_ids = mx.nd.array([[0, 0, 1], [0, 1, 0]])
>>> model, vocab = get_mxnet_kobert_model(use_decoder=False, use_classifier=False)
>>> encoder_layer, pooled_output = model(input_id, token_type_ids)
>>> pooled_output.shape
(2, 768)
>>> vocab
Vocab(size=8002, unk="[UNK]", reserved="['[MASK]', '[SEP]', '[CLS]']")
>>> # Last Encoding Layer
>>> encoder_layer[0]
[[-0.24610372  0.24282135  0.2589539  ... -0.48613444 -0.07305248
   0.07560539]
 [-0.24783105  0.242005    0.25520545 ... -0.48771808 -0.07270523
   0.07536077]
 [-0.24721491  0.241966    0.25606337 ... -0.48743105 -0.07327032
   0.07650219]]
<NDArray 3x768 @cpu(0)>
```

- Naver Sentiment Analysis Fine-Tuning with MXNet
  - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_gluon_kobert.ipynb)

#### Tokenizer

* Pretrained [Sentencepiece](https://github.com/google/sentencepiece) tokenizer

```python
>>> from gluonnlp.data import SentencepieceTokenizer
>>> from kobert.utils import get_tokenizer
>>> tok_path = get_tokenizer()
>>> sp  = SentencepieceTokenizer(tok_path)
>>> sp('한국어 모델을 공유합니다.')
['▁한국', '어', '▁모델', '을', '▁공유', '합니다', '.']
```

---

### Subtasks

#### Naver Sentiment Analysis

- Dataset : https://github.com/e9t/nsmc


| Model |  Accuracy  |
|---|---|
| [BERT base multilingual cased](https://github.com/google-research/bert/blob/master/multilingual.md) |  0.875  |
| KoBERT | **[0.901](logs/bert_naver_small_512_news_simple_20190624.txt)**|
| [KoGPT2](https://github.com/SKT-AI/KoGPT2) | 0.899 |

#### KoBERT와 CRF로 만든 한국어 객체명인식기

- https://github.com/eagle705/pytorch-bert-crf-ner


```
문장을 입력하세요:  SKTBrain에서 KoBERT 모델을 공개해준 덕분에 BERT-CRF 기반 객체명인식기를 쉽게 개발할 수 있었다.
len: 40, input_token:['[CLS]', '▁SK', 'T', 'B', 'ra', 'in', '에서', '▁K', 'o', 'B', 'ER', 'T', '▁모델', '을', '▁공개', '해', '준', '▁덕분에', '▁B', 'ER', 'T', '-', 'C', 'R', 'F', '▁기반', '▁', '객', '체', '명', '인', '식', '기를', '▁쉽게', '▁개발', '할', '▁수', '▁있었다', '.', '[SEP]']
len: 40, pred_ner_tag:['[CLS]', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'B-POH', 'I-POH', 'I-POH', 'I-POH', 'I-POH', 'O', 'O', 'O', 'O', 'O', 'O', 'B-POH', 'I-POH', 'I-POH', 'I-POH', 'I-POH', 'I-POH', 'I-POH', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', '[SEP]']
decoding_ner_sentence: [CLS] <SKTBrain:ORG>에서 <KoBERT:POH> 모델을 공개해준 덕분에 <BERT-CRF:POH> 기반 객체명인식기를 쉽게 개발할 수 있었다.[SEP]
```

---

### Version History

* v.0.1 : 초기 모델 릴리즈
* v.0.1.1 : 사전(vocabulary)과 토크나이저 통합

### Contacts

`KoBERT` 관련 이슈는 [이곳](https://github.com/SKTBrain/KoBERT/issues)에 등록해 주시기 바랍니다.

### License

`KoBERT`는 Apache-2.0 라이선스 하에 공개되어 있습니다. 모델 및 코드를 사용할 경우 라이선스 내용을 준수해주세요. 라이선스 전문은 `LICENSE` 파일에서 확인하실 수 있습니다.
