import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import boto3
from botocore.config import Config
from p_tqdm import p_imap
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm


class RemoteStore(ABC):
    @abstractmethod
    def upload_folder(self, local_folder: Path | str, bucket_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def upload_file(
        self,
        file_path: str | Path,
        bucket_name: str,
        output_file_name: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def download_folder(
        self,
        local_folder: str | Path,
        bucket_name: str,
        predicate: Callable | None = None,
    ) -> None:
        raise NotImplementedError


class S3RemoteStore(RemoteStore):
    def __init__(self, url: str, key_id: str, secret: str) -> None:
        self.url = url
        self.key_id = key_id
        self.secret = secret

        self.api = boto3.client(
            "s3",
            endpoint_url=url,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            config=Config(
                signature_version="s3v4",
            ),
        )
        self.resource = boto3.resource(
            "s3",
            endpoint_url=url,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            config=Config(
                signature_version="s3v4",
            ),
        )

    def upload_folder(self, local_folder: Path | str, bucket_name: str) -> None:
        local_folder = Path(local_folder)
        for file in tqdm(os.listdir(local_folder)):
            target_file_name = local_folder.joinpath(file)
            self.api.upload_file(target_file_name, bucket_name, file)

    def upload_file(
        self,
        file_path: Path,
        bucket_name: str,
        output_file_name: str | None = None,
    ) -> None:
        if output_file_name is None:
            output_file_name = file_path.name
        self.api.upload_file(file_path, bucket_name, output_file_name)

    def download_folder(
        self,
        local_folder: str | Path,
        bucket_name: str,
        predicate: Callable | None = None,
    ) -> None:
        local_folder = Path(local_folder)
        local_folder.mkdir(parents=True, exist_ok=True)

        bucket = self.resource.Bucket(bucket_name)

        urls = [
            (
                obj.key,
                bucket.meta.client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": bucket_name, "Key": obj.key},
                ),
            )
            for obj in bucket.objects.all()
        ]
        if predicate is not None:
            urls = list(filter(predicate, urls))

        def download(tup):
            @retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
            def execute():
                obj_key, url = tup
                return_code = os.system(
                    f"wget '{url}' -O '{local_folder.joinpath(obj_key)}.tmp' -q",
                )
                os.system(
                    f"mv '{local_folder.joinpath(obj_key)}.tmp' '{local_folder.joinpath(obj_key)}'"
                )
                if return_code != 0:
                    raise RuntimeError(f"Failed to download {url}")

            return execute()

        return_codes = p_imap(download, urls)
        if any(return_codes):
            raise RuntimeError(f"Failed to download files from {bucket_name}")

    def delete_all_files(self, bucket_name: str) -> None:
        bucket = self.resource.Bucket(bucket_name)
        bucket.objects.all().delete()


def download(tup: tuple[str, str, str]):
    header = tup[0]
    url = tup[1]
    output_path = tup[2]

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(10))
    def execute():
        return_code = os.system(f"curl -L -o {output_path} '{url}' -H '{header}'")
        if return_code != 0:
            raise RuntimeError(f"Failed to download {url}")

    return execute()
