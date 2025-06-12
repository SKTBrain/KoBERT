# coding=utf-8
# Copyright 2019-2025 SKTelecom
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

import hashlib
import os
import urllib.request


def download(url, chksum=None, cachedir=".cache"):
    cachedir_full = os.path.join(os.getcwd(), cachedir)
    os.makedirs(cachedir_full, exist_ok=True)
    filename = os.path.basename(url)
    file_path = os.path.join(cachedir_full, filename)
    if os.path.isfile(file_path):
        if hashlib.md5(open(file_path, "rb").read()).hexdigest()[:10] == chksum[:10]:
            print(f"using cached model. {file_path}")
            return file_path, True

    print(f"downloading model from {url}...")
    try:
        urllib.request.urlretrieve(url, file_path)
    except Exception as e:
        print(f"download failed: {e}")
        return None, False

    if chksum:
        assert (
            chksum[:10] == hashlib.md5(open(file_path, "rb").read()).hexdigest()[:10]
        ), "corrupted file!"
    return file_path, False


def get_tokenizer_path(cachedir=".cache"):
    """Get KoBERT Tokenizer file path after downloading"""
    tokenizer = {
        "url": "https://huggingface.co/skt/kobert-base-v1/resolve/main/legacy/kobert_news_wiki_ko_cased-1087f8699e.spiece",
        "chksum": "ae5711deb3",
    }

    model_info = tokenizer
    model_path, _ = download(model_info["url"], model_info["chksum"], cachedir=cachedir)
    return model_path
