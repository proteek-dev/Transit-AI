"""
SEQ TransLink GTFS-Realtime Archiver
Fetches trip_updates, vehicle_positions, service_alerts every 5 minutes and uploads to S3.
Downloads static GTFS once per day. All data written directly to Amazon S3.
"""

import io
import json
import os
import sys
import time
import zipfile
from datetime import datetime, date
from pathlib import Path

import boto3
import requests
import yaml
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from google.transit import gtfs_realtime_pb2

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "feeds.yaml"

# ── AWS ───────────────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_S3_BUCKET = os.environ["AWS_S3_BUCKET"]
AWS_REGION = os.environ["AWS_REGION"]

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# ── Config ────────────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

FEEDS = CONFIG["gtfs_realtime"]
STATIC_URL = CONFIG["gtfs_static"]["seq"]
FETCH_INTERVAL = CONFIG["fetch_interval_seconds"]

# TransLink's combined SEQ/TripUpdates feed does not include tram entities —
# tram is only published via this dedicated per-mode endpoint. Polled
# separately, alongside the combined feed, in the main loop below.
TRAM_TRIP_UPDATES_URL = CONFIG.get("gtfs_realtime_by_mode", {}).get("trip_updates_tram")


def log(feed: str, record_count: int, s3_key: str, status: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {feed:<20} | records={record_count:<6} | {s3_key} | {status}")


# ── Protobuf parsers ──────────────────────────────────────────────────────────

def parse_trip_updates(feed_message) -> list[dict]:
    records = []
    ts = int(datetime.now().timestamp())
    for entity in feed_message.entity:
        if not entity.HasField("trip_update"):
            continue
        tu = entity.trip_update
        trip_id = tu.trip.trip_id
        route_id = tu.trip.route_id
        for stu in tu.stop_time_update:
            records.append({
                "trip_id": trip_id,
                "route_id": route_id,
                "stop_id": stu.stop_id,
                "delay_seconds": stu.arrival.delay if stu.HasField("arrival") else None,
                "timestamp": ts,
            })
    return records


def parse_vehicle_positions(feed_message) -> list[dict]:
    records = []
    ts = int(datetime.now().timestamp())
    for entity in feed_message.entity:
        if not entity.HasField("vehicle"):
            continue
        vp = entity.vehicle
        records.append({
            "vehicle_id": vp.vehicle.id,
            "trip_id": vp.trip.trip_id,
            "lat": vp.position.latitude,
            "lon": vp.position.longitude,
            "bearing": vp.position.bearing if vp.position.HasField("bearing") else None,
            "timestamp": ts,
        })
    return records


def parse_service_alerts(feed_message) -> list[dict]:
    records = []
    ts = int(datetime.now().timestamp())
    for entity in feed_message.entity:
        if not entity.HasField("alert"):
            continue
        alert = entity.alert
        cause = gtfs_realtime_pb2.Alert.Cause.Name(alert.cause)
        effect = gtfs_realtime_pb2.Alert.Effect.Name(alert.effect)
        header = alert.header_text.translation[0].text if alert.header_text.translation else ""
        desc = alert.description_text.translation[0].text if alert.description_text.translation else ""
        informed = [
            {"route_id": ie.route_id, "stop_id": ie.stop_id, "trip_id": ie.trip.trip_id}
            for ie in alert.informed_entity
        ]
        records.append({
            "alert_id": entity.id,
            "cause": cause,
            "effect": effect,
            "header": header,
            "description": desc,
            "informed_entities": informed,
            "timestamp": ts,
        })
    return records


PARSERS = {
    "trip_updates": parse_trip_updates,
    "vehicle_positions": parse_vehicle_positions,
    "service_alerts": parse_service_alerts,
}


# ── Fetch + save ──────────────────────────────────────────────────────────────

def fetch_and_save(feed_name: str, url: str) -> int:
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    s3_key = f"gtfs_realtime/{feed_name}/{date_str}/{time_str}.json"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        feed_message = gtfs_realtime_pb2.FeedMessage()
        feed_message.ParseFromString(resp.content)

        records = PARSERS[feed_name](feed_message)
        body = json.dumps(records, separators=(",", ":")).encode()

        s3.put_object(Bucket=AWS_S3_BUCKET, Key=s3_key, Body=body)
        log(feed_name, len(records), s3_key, "OK")
        return len(records)

    except Exception as e:
        log(feed_name, 0, s3_key, f"ERROR: {e}")
        return 0


def fetch_and_save_tram(url: str) -> int:
    """Belt-and-suspenders poll of the dedicated TripUpdates/Tram endpoint —
    the combined feed above doesn't carry tram entities. Same fetch/parse/save
    pattern as fetch_and_save(), archived alongside the combined trip_updates
    output (same date folder, "_tram" filename suffix) rather than a separate
    prefix, and never crashes the daemon on failure or an empty response.
    """
    if not url:
        log("trip_updates_tram", 0, "-", "WARNING: no tram endpoint configured in feeds.yaml")
        return 0

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    s3_key = f"gtfs_realtime/trip_updates/{date_str}/{time_str}_tram.json"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        feed_message = gtfs_realtime_pb2.FeedMessage()
        feed_message.ParseFromString(resp.content)

        records = parse_trip_updates(feed_message)
        body = json.dumps(records, separators=(",", ":")).encode()

        s3.put_object(Bucket=AWS_S3_BUCKET, Key=s3_key, Body=body)
        log("trip_updates_tram", len(records), s3_key, "OK")
        return len(records)

    except Exception as e:
        log("trip_updates_tram", 0, s3_key, f"WARNING: {e}")
        return 0


# ── Static GTFS ───────────────────────────────────────────────────────────────

def maybe_download_static_gtfs() -> None:
    today = date.today().strftime("%Y-%m-%d")
    prefix = f"gtfs_static/{today}/"

    existing = s3.list_objects_v2(Bucket=AWS_S3_BUCKET, Prefix=prefix, MaxKeys=1)
    if existing.get("KeyCount", 0) > 0:
        return

    print(f"\n[static] Downloading SEQ_GTFS.zip for {today} ...")
    try:
        resp = requests.get(STATIC_URL, timeout=120, stream=True)
        resp.raise_for_status()

        zip_bytes = io.BytesIO(resp.content)
        with zipfile.ZipFile(zip_bytes) as z:
            for name in z.namelist():
                s3_key = f"gtfs_static/{today}/{name}"
                s3.put_object(Bucket=AWS_S3_BUCKET, Key=s3_key, Body=z.read(name))
                print(f"[static] Uploaded {s3_key}")

        print(f"[static] Extracted to s3://{AWS_S3_BUCKET}/{prefix} OK")
    except Exception as e:
        print(f"[static] ERROR downloading static GTFS: {e}")


# ── Performance CSV downloader ────────────────────────────────────────────────

def download_performance_data(config) -> None:
    """
    Downloads all performance CSVs defined in config/feeds.yaml.
    Only re-downloads if object is older than 24 hours.
    Called once at startup, then every 24 hours in the main loop.
    """
    csvs = config.get("performance_csvs", {})

    if not csvs:
        print("[performance] No performance_csvs found in config. Skipping.")
        return

    for key, meta in csvs.items():
        url = meta.get("url", "")
        filename = meta.get("filename", f"{key}.csv")
        s3_key = f"performance/{filename}"

        try:
            head = s3.head_object(Bucket=AWS_S3_BUCKET, Key=s3_key)
            age_hours = (time.time() - head["LastModified"].timestamp()) / 3600
            if age_hours < 24:
                print(f"[performance] SKIP {filename} — uploaded {age_hours:.1f}h ago")
                continue
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                print(f"[performance] FAIL {filename} — head_object error: {e}")
                continue

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            s3.put_object(Bucket=AWS_S3_BUCKET, Key=s3_key, Body=resp.content)
            row_count = len(resp.text.strip().splitlines()) - 1
            print(f"[performance] OK   {filename} — {row_count} rows — s3://{AWS_S3_BUCKET}/{s3_key}")
        except Exception as e:
            print(f"[performance] FAIL {filename} — {url} — {e}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("SEQ TransLink GTFS-Realtime Archiver")
    print("=" * 70)
    print(f"  Feeds  : {', '.join(FEEDS.keys())}")
    print(f"  Source : {', '.join(FEEDS.values())}")
    print(f"  Fetch  : every {FETCH_INTERVAL}s ({FETCH_INTERVAL // 60} min)")
    print(f"  Output : s3://{AWS_S3_BUCKET}/gtfs_realtime/")
    print(f"  Static : s3://{AWS_S3_BUCKET}/gtfs_static/ (once per day)")
    print("=" * 70)
    print()

    maybe_download_static_gtfs()

    print("\n=== Downloading performance CSVs at startup ===")
    download_performance_data(CONFIG)
    print("=== Performance download complete ===\n")

    while True:
        for feed_name, url in FEEDS.items():
            fetch_and_save(feed_name, url)

        fetch_and_save_tram(TRAM_TRIP_UPDATES_URL)

        download_performance_data(CONFIG)

        time.sleep(FETCH_INTERVAL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SEQ TransLink GTFS-Realtime Archiver")
    parser.add_argument(
        "--download-performance-only",
        action="store_true",
        help="Download performance CSVs once and exit (no GTFS-RT loop)",
    )
    args = parser.parse_args()

    if args.download_performance_only:
        print("=== Downloading performance CSVs (one-off mode) ===")
        download_performance_data(CONFIG)
        print("=== Done ===")
        sys.exit(0)

    try:
        main()
    except KeyboardInterrupt:
        print("\n[archiver] Stopped by user.")
        sys.exit(0)
