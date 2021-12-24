import boto3
import os
import sys
from botocore import UNSIGNED
from botocore.client import Config


class AwsS3Downloader(object):
    def __init__(
        self,
        aws_access_key_id=None,
        aws_secret_access_key=None,
    ):
        self.resource = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        ).resource("s3")
        self.client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            config=Config(signature_version=UNSIGNED),
        )

    def __split_url(self, url: str):
        if url.startswith("s3://"):
            url = url.replace("s3://", "")
        bucket, key = url.split("/", maxsplit=1)
        return bucket, key

    def download(self, url: str, local_dir: str):
        bucket, key = self.__split_url(url)
        filename = os.path.basename(key)
        file_path = os.path.join(local_dir, filename)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        meta_data = self.client.head_object(Bucket=bucket, Key=key)
        total_length = int(meta_data.get("ContentLength", 0))

        downloaded = 0

        def progress(chunk):
            nonlocal downloaded
            downloaded += chunk
            done = int(50 * downloaded / total_length)
            sys.stdout.write(
                "\r{}[{}{}]".format(file_path, "█" * done, "." * (50 - done))
            )
            sys.stdout.flush()

        try:
            with open(file_path, "wb") as f:
                self.client.download_fileobj(bucket, key, f, Callback=progress)
            sys.stdout.write("\n")
            sys.stdout.flush()
        except:
            raise Exception(f"downloading file is failed. {url}")
        return file_path


if __name__ == "__main__":
    s3 = AwsS3Downloader()

    s3.download(
        url="s3://skt-lsl-nlp-model/KoBERT/tokenizers/kobert_news_wiki_ko_cased-1087f8699e.spiece",
        local_dir=".cache",
    )
