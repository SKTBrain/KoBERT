

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Korean BERT pre-trained cased (KoBERT) for Huggingface Transformers](#korean-bert-pre-trained-cased-kobert-for-huggingface-transformers)
  - [Requirements](#requirements)
  - [How to install](#how-to-install)
- [Tokenizer](#tokenizer)
- [Model](#model)
- [License](#license)

<!-- /code_chunk_output -->

---

### Korean BERT pre-trained cased (KoBERT) for Huggingface Transformers

KoBERT를 Huggingface.co 기반으로 사용할 수 있게 Wrapping 작업을 수행하였습니다.


#### Requirements

* Python >= 3.6
* PyTorch >= 1.8.1
* transformers >= 4.8.2

#### How to install

```sh
git clone https://github.com/SKTBrain/KoBERT.git
cd KoBERT/kobert_hf
pip install -r requirements.txt
python setup.py install
```

---

### Tokenizer

[XLNetTokenizer](https://github.com/huggingface/transformers/blob/master/src/transformers/models/xlnet/tokenization_xlnet.py)를 활용하여 Wrapping 작업을 진행하였습니다.

기존 Tokenizer와 동일하게 사전 크기는 8,002개 입니다.

```python
> from kobert_tokenizer import KoBERTTokenizer
> tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
> tokenizer.encode("한국어 모델을 공유합니다.")
[2, 4958, 6855, 2046, 7088, 1050, 7843, 54, 3]

```

### Model
```python
> import torch
> from transformers import BertModel
> model = BertModel.from_pretrained('skt/kobert-base-v1')
> text = "한국어 모델을 공유합니다."
> inputs = tokenizer.batch_encode_plus([text])
> out = model(input_ids = torch.tensor(inputs['input_ids']),
              token_type_ids = torch.tensor(inputs['token_type_ids']),
              attention_mask = torch.tensor(inputs['attention_mask']))
> out.pooler_output.shape
torch.Size([1, 768])

```

### License

`KoBERT`는 Apache-2.0 라이선스 하에 공개되어 있습니다. 모델 및 코드를 사용할 경우 라이선스 내용을 준수해주세요. 라이선스 전문은 `LICENSE` 파일에서 확인하실 수 있습니다.
