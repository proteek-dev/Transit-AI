"""Shared config helpers for the Phase 3 Streamlit POC.

Centralizes AWS credential loading so every module works unmodified in both
environments: localhost (reads .env) and Streamlit Community Cloud (reads
st.secrets, which is the only place Cloud deployments can put credentials —
there is no server shell to export env vars into).
"""
from __future__ import annotations

import os

import s3fs


def get_aws_credentials() -> dict:
    """Try Streamlit secrets first, then env vars, then .env file."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'AWS_ACCESS_KEY_ID' in st.secrets:
            return {
                'aws_access_key_id': st.secrets['AWS_ACCESS_KEY_ID'],
                'aws_secret_access_key': st.secrets['AWS_SECRET_ACCESS_KEY'],
                'region_name': st.secrets.get('AWS_REGION', 'ap-southeast-2'),
            }
    except Exception:
        pass

    # Fall back to env vars / .env
    from dotenv import find_dotenv, load_dotenv
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)

    return {
        'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'region_name': os.getenv('AWS_REGION', 'ap-southeast-2'),
    }


def get_s3_bucket() -> str:
    """Same secrets-first, then env/.env fallback, for the S3 bucket name."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'AWS_S3_BUCKET' in st.secrets:
            return st.secrets['AWS_S3_BUCKET']
    except Exception:
        pass

    from dotenv import find_dotenv, load_dotenv
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)

    bucket = os.getenv('AWS_S3_BUCKET', '')
    if not bucket:
        raise EnvironmentError('AWS_S3_BUCKET is not set. Check your .env file or Streamlit secrets.')
    return bucket


def get_s3_filesystem() -> s3fs.S3FileSystem:
    """One S3FileSystem built from get_aws_credentials(), for any module that needs S3 access."""
    creds = get_aws_credentials()
    return s3fs.S3FileSystem(
        key=creds['aws_access_key_id'],
        secret=creds['aws_secret_access_key'],
        client_kwargs={'region_name': creds['region_name']},
    )
