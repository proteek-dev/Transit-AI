"""
SEQ TransLink GTFS-Realtime Archiver
Fetches trip_updates, vehicle_positions, service_alerts every 5 minutes and archives to JSON.
Commits to git every 30 minutes. Downloads static GTFS once per day.
"""

import json
import os
import sys
import time
import zipfile
from datetime import datetime, date
from pathlib import Path

import requests
import yaml
from google.transit import gtfs_realtime_pb2

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "feeds.yaml"
GTFSRT_DIR = REPO_ROOT / "source_files" / "gtfs_realtime"
STATIC_DIR = REPO_ROOT / "source_files" / "gtfs_static"

# ── Config ────────────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

FEEDS = CONFIG["gtfs_realtime"]
STATIC_URL = CONFIG["gtfs_static"]["seq"]
FETCH_INTERVAL = CONFIG["fetch_interval_seconds"]
COMMIT_INTERVAL = 1800  # 30 minutes


def log(feed: str, record_count: int, file_path: str, status: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | {feed:<20} | records={record_count:<6} | {file_path} | {status}")


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

    out_dir = GTFSRT_DIR / feed_name / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{time_str}.json"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()

        feed_message = gtfs_realtime_pb2.FeedMessage()
        feed_message.ParseFromString(resp.content)

        records = PARSERS[feed_name](feed_message)

        with open(out_path, "w") as f:
            json.dump(records, f, separators=(",", ":"))

        log(feed_name, len(records), str(out_path.relative_to(REPO_ROOT)), "OK")
        return len(records)

    except Exception as e:
        log(feed_name, 0, str(out_path.relative_to(REPO_ROOT)), f"ERROR: {e}")
        return 0


# ── Static GTFS ───────────────────────────────────────────────────────────────

def maybe_download_static_gtfs() -> None:
    today = date.today().strftime("%Y-%m-%d")
    today_dir = STATIC_DIR / today
    if today_dir.exists():
        return

    print(f"\n[static] Downloading SEQ_GTFS.zip for {today} ...")
    zip_path = STATIC_DIR / "SEQ_GTFS.zip"
    try:
        resp = requests.get(STATIC_URL, timeout=120, stream=True)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

        today_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(today_dir)
        zip_path.unlink()
        print(f"[static] Extracted to {today_dir.relative_to(REPO_ROOT)} OK")
    except Exception as e:
        print(f"[static] ERROR downloading static GTFS: {e}")


# ── Git commit ────────────────────────────────────────────────────────────────

def git_commit() -> None:
    now = datetime.now()
    label = now.strftime("%Y-%m-%d %H:%M")
    cmd = (
        f'cd "{REPO_ROOT}" && '
        f'git add source_files/ && '
        f'git diff --cached --quiet || ('
        f'git commit -m "archive: {label} | trip_updates + vehicle_positions + alerts" && '
        f'git push origin work_ai'
        f')'
    )
    ret = os.system(cmd)
    if ret == 0:
        print(f"[git] Committed and pushed at {label}")
    else:
        print(f"[git] Commit/push failed (exit {ret}) — will retry next cycle")


# ── Performance CSV downloader ────────────────────────────────────────────────

def download_performance_data(config, output_dir="source_files/performance") -> None:
    """
    Downloads all performance CSVs defined in config/feeds.yaml.
    Only re-downloads if file is older than 24 hours.
    Called once at startup, then every 24 hours in the main loop.
    """
    out_path = REPO_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    csvs = config.get("performance_csvs", {})

    if not csvs:
        print("[performance] No performance_csvs found in config. Skipping.")
        return

    for key, meta in csvs.items():
        url = meta.get("url", "")
        filename = meta.get("filename", f"{key}.csv")
        filepath = out_path / filename

        if filepath.exists():
            age_hours = (time.time() - filepath.stat().st_mtime) / 3600
            if age_hours < 24:
                print(f"[performance] SKIP {filename} — downloaded {age_hours:.1f}h ago")
                continue

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            filepath.write_bytes(resp.content)
            row_count = len(resp.text.strip().splitlines()) - 1
            print(f"[performance] OK   {filename} — {row_count} rows — saved to {filepath.relative_to(REPO_ROOT)}")
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
    print(f"  Commit : every {COMMIT_INTERVAL}s ({COMMIT_INTERVAL // 60} min)")
    print(f"  Output : {GTFSRT_DIR.relative_to(REPO_ROOT)}")
    print(f"  Static : {STATIC_DIR.relative_to(REPO_ROOT)} (once per day)")
    print("=" * 70)
    print()

    last_commit_time = 0.0

    maybe_download_static_gtfs()

    print("\n=== Downloading performance CSVs at startup ===")
    download_performance_data(CONFIG)
    print("=== Performance download complete ===\n")

    while True:
        for feed_name, url in FEEDS.items():
            fetch_and_save(feed_name, url)

        now = time.time()
        if now - last_commit_time >= COMMIT_INTERVAL:
            git_commit()
            last_commit_time = now

        # Re-check performance CSVs every 24 hours (skips if fresh)
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
