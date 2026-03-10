"""
s3_helper.py – thin wrapper around boto3 for S3 uploads.

Required environment variables (set in .env or your deployment config):
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_REGION
    AWS_BUCKET_NAME
"""

import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

load_dotenv()

_AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
_AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
_AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

if not _BUCKET_NAME:
    raise ValueError("AWS_BUCKET_NAME environment variable is not set")

# Build a single reusable client.  boto3 clients are thread-safe for reads;
# uploads use distinct TCP connections so sharing one client across threads
# is safe here.
_s3_client = boto3.client(
    "s3",
    aws_access_key_id=_AWS_ACCESS_KEY_ID,
    aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
    region_name=_AWS_REGION,
)


def upload_bytes_to_s3(
    data: bytes,
    s3_key: str,
    content_type: str = "image/jpeg",
) -> str:
    """
    Upload raw bytes to S3 under `s3_key` and return the key.

    Raises RuntimeError on upload failure so callers can surface the error
    without importing boto3 exceptions directly.
    """
    try:
        _s3_client.put_object(
            Bucket=_BUCKET_NAME,
            Key=s3_key,
            Body=data,
            ContentType=content_type,
        )
        return s3_key
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"S3 upload failed for key '{s3_key}': {exc}") from exc


def upload_image_to_s3(
    file_path: str,
    s3_key: str,
    content_type: str = "image/jpeg",
) -> str:
    """
    Upload a file from disk to S3 and return the key.
    Provided for convenience; the main flow uses upload_bytes_to_s3.
    """
    try:
        _s3_client.upload_file(
            Filename=file_path,
            Bucket=_BUCKET_NAME,
            Key=s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        return s3_key
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(
            f"S3 upload failed for file '{file_path}' → key '{s3_key}': {exc}"
        ) from exc


def generate_presigned_url(s3_key: str, expires_in: int = 3600) -> str:
    """
    Return a pre-signed URL that lets the frontend fetch the image directly
    from S3 without exposing credentials.  Valid for `expires_in` seconds
    (default 1 hour).
    """
    try:
        return _s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": _BUCKET_NAME, "Key": s3_key},
            ExpiresIn=expires_in,
        )
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(
            f"Failed to generate pre-signed URL for '{s3_key}': {exc}"
        ) from exc