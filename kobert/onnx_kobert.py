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

from kobert.utils import download as _download


def get_onnx_kobert_model(cachedir=".cache"):
    """Get KoBERT ONNX file path after downloading"""
    onnx_kobert = {
        "url": "https://kobert.blob.core.windows.net/models/kobert/onnx/onnx_kobert_44529811f0.onnx",
        "fname": "onnx_kobert_44529811f0.onnx",
        "chksum": "44529811f0",
    }

    model_info = onnx_kobert
    return _download(
        model_info["url"], model_info["fname"], model_info["chksum"], cachedir=cachedir
    )


if __name__ == "__main__":
    import onnxruntime
    import numpy as np
    from kobert import get_onnx_kobert_model

    onnx_path = get_onnx_kobert_model()
    sess = onnxruntime.InferenceSession(onnx_path)
    input_ids = [[31, 51, 99], [15, 5, 0]]
    input_mask = [[1, 1, 1], [1, 1, 0]]
    token_type_ids = [[0, 0, 1], [0, 1, 0]]
    len_seq = len(input_ids[0])
    pred_onnx = sess.run(
        None,
        {
            "input_ids": np.array(input_ids),
            "token_type_ids": np.array(token_type_ids),
            "input_mask": np.array(input_mask),
            "position_ids": np.array(range(len_seq)),
        },
    )
    print(pred_onnx[-2][0])
