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

from kobert import download


def get_onnx_kobert_model(cachedir=".cache"):
    """Get KoBERT ONNX file path after downloading"""
    onnx_kobert = {
        "url": "s3://skt-lsl-nlp-model/KoBERT/models/kobert.onnx1.8.0.onnx",
        "chksum": "6f6610f2e3b61da6de8dbce",
    }

    model_info = onnx_kobert
    model_path, is_cached = download(
        model_info["url"], model_info["chksum"], cachedir=cachedir
    )
    return model_path


def make_dummy_input(max_seq_len):
    def do_pad(x, max_seq_len, pad_id):
        return [_x + [pad_id] * (max_seq_len - len(_x)) for _x in x]

    input_ids = do_pad([[31, 51, 99], [15, 5]], max_seq_len, pad_id=1)
    token_type_ids = do_pad([[0, 0, 0], [0, 0]], max_seq_len, pad_id=0)
    input_mask = do_pad([[1, 1, 1], [1, 1]], max_seq_len, pad_id=0)
    position_ids = list(range(max_seq_len))
    return (input_ids, token_type_ids, input_mask, position_ids)


if __name__ == "__main__":
    import onnxruntime
    import numpy as np
    from kobert import get_onnx_kobert_model

    onnx_path = get_onnx_kobert_model()
    dummy_input = make_dummy_input(max_seq_len=512)
    so = onnxruntime.SessionOptions()
    sess = onnxruntime.InferenceSession(onnx_path)
    outputs = sess.run(
        None,
        {
            "input_ids": np.array(dummy_input[0]),
            "token_type_ids": np.array(dummy_input[1]),
            "input_mask": np.array(dummy_input[2]),
            "position_ids": np.array(dummy_input[3]),
        },
    )
    print(outputs[-2][0])
