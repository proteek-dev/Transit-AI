# Run once to backfill existing local data to S3. Safe to re-run.
# Requires boto3: pip install boto3

import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION")

missing = [
    name
    for name, val in [
        ("AWS_ACCESS_KEY_ID", AWS_ACCESS_KEY_ID),
        ("AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY),
        ("AWS_S3_BUCKET", AWS_S3_BUCKET),
        ("AWS_REGION", AWS_REGION),
    ]
    if not val
]
if missing:
    print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
    sys.exit(1)

LOCAL_DATA_DIR = Path(
    os.environ.get("TRANSIT_AI_DATA_DIR", "~/transit-ai-data")
).expanduser()

if not LOCAL_DATA_DIR.exists():
    print(f"ERROR: LOCAL_DATA_DIR does not exist: {LOCAL_DATA_DIR}")
    sys.exit(1)

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

attempted = 0
succeeded = 0
failed = 0

for dirpath, _dirnames, filenames in os.walk(LOCAL_DATA_DIR):
    dir_rel = Path(dirpath).relative_to(LOCAL_DATA_DIR)
    if dir_rel.parts and dir_rel.parts[0] == "logs":
        continue

    for filename in filenames:
        if filename.startswith("."):
            continue

        local_path = Path(dirpath) / filename
        s3_key = str(Path(dirpath).relative_to(LOCAL_DATA_DIR) / filename)

        attempted += 1
        try:
            s3.upload_file(str(local_path), AWS_S3_BUCKET, s3_key)
            print(f"Uploaded: {s3_key}")
            succeeded += 1
        except (BotoCoreError, ClientError, OSError) as e:
            print(f"FAILED: {s3_key} — {e}")
            failed += 1

print(f"\nDone. Attempted: {attempted}, Succeeded: {succeeded}, Failed: {failed}")
