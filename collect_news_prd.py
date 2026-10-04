"""Bounded, resumable Alpha Vantage collection. Default: plan only, no API calls."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import itertools
import json
import os
import re
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import pyarrow.parquet as pq
import yaml

URL = "https://www.alphavantage.co/query"
LIMIT = 1000
COLUMNS = ["ticker", "time_published", "title", "summary", "source", "url"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def quarters(start, end):
    while start <= end:
        following = (pd.Timestamp(start) + pd.DateOffset(months=3)).date()
        last = min(end, following - timedelta(days=1))
        yield start, last
        start = last + timedelta(days=1)


def subtract(start, end, covered):
    cursor = start
    for lo, hi in sorted(covered):
        if hi < cursor or lo > end:
            continue
        if lo > cursor:
            yield cursor, min(end, lo - timedelta(days=1))
        cursor = max(cursor, hi + timedelta(days=1))
        if cursor > end:
            return
    if cursor <= end:
        yield cursor, end


def job_key(ticker, start, end):
    return f"{ticker}_{start:%Y%m%d}_{end:%Y%m%d}"


def legacy_inventory(root, ticker):
    covered, details = [], []
    for path in sorted((root / "data/raw/news" / ticker).glob("*.parquet")):
        match = re.fullmatch(re.escape(ticker) + r"_(\d{8})_(\d{8})\.parquet", path.name)
        if not match:
            continue
        lo, hi = [datetime.strptime(v, "%Y%m%d").date() for v in match.groups()]
        rows = pq.read_metadata(path).num_rows
        if lo > hi:
            raise ValueError(f"Invalid filename date range: {path.name}")
        if 0 < rows < LIMIT:
            covered.append((lo, hi))
        details.append(
            {
                "file": str(path.relative_to(root)),
                "rows": rows,
                "status": "legacy_unverified" if 0 < rows < LIMIT else "needs_review",
            }
        )
    return covered, details


def children(ticker, start, end):
    if start == end:
        raise ValueError(
            f"{job_key(ticker, start, end)} reached the cap in one day; manual review required."
        )
    middle = start + timedelta(days=(end - start).days // 2)
    return [(ticker, start, middle), (ticker, middle + timedelta(days=1), end)]


def pending(task, records, audit, root):
    ticker, start, end = task
    key = job_key(*task)
    record = records.get(key)
    if not record or record["status"] in ("error", "requesting"):
        return [task]
    for name, sha in record.get("files", {}).items():
        path = root / name
        if not path.is_file() or digest(path) != sha:
            raise ValueError(f"Saved collection file changed or missing: {name}")
    if record["status"] == "capped":
        return list(
            itertools.chain.from_iterable(pending(t, records, audit, root) for t in children(*task))
        )
    if record["status"] not in ("retrieved_below_cap", "empty_unverified"):
        raise ValueError(f"Unknown collection status for {key}")
    return []


def make_plan(root, config, tickers, records, audit):
    start = date.fromisoformat(config["start_date"])
    end = date.fromisoformat(config["regression"]["end_date"])
    if start > end:
        raise ValueError("Configured start date is after end date.")
    queues, inventory = [], []
    for ticker in tickers:
        covered, details = legacy_inventory(root, ticker)
        inventory += details
        jobs = [
            (ticker, a, b) for lo, hi in quarters(start, end) for a, b in subtract(lo, hi, covered)
        ]
        queues.append(
            list(itertools.chain.from_iterable(pending(t, records, audit, root) for t in jobs))
        )
    tasks = [t for row in itertools.zip_longest(*queues) for t in row if t is not None]
    return tasks, inventory


def validate_response(data, task):
    ticker, start, end = task
    if not isinstance(data, dict):
        raise ValueError("Provider response is not a JSON object.")
    for key in ("Error Message", "Information", "Note"):
        if key in data:
            raise ValueError(f"Provider returned {key}: {str(data[key])[:400]}")
    feed = data.get("feed")
    if not isinstance(feed, list):
        raise ValueError("Provider response has no valid feed; this is not a no-news result.")
    if "items" in data and int(data["items"]) != len(feed):
        raise ValueError("Provider item count differs from feed length.")
    rows = []
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end + timedelta(days=1), tz="UTC")
    for item in feed:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("title"), str)
            or not item["title"].strip()
        ):
            raise ValueError("An article lacks a valid title; response needs review.")
        stamp = pd.to_datetime(
            item.get("time_published"), format="%Y%m%dT%H%M%S", utc=True, errors="raise"
        )
        if pd.isna(stamp) or stamp < lo or stamp > hi:
            raise ValueError("An article timestamp falls outside the requested interval.")
        if stamp == hi:
            continue
        rows.append(
            {
                "ticker": ticker,
                "time_published": stamp,
                "title": item["title"],
                **{k: item.get(k) or "" for k in ("summary", "source", "url")},
            }
        )
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates(["ticker", "time_published", "title"]).sort_values(
            "time_published"
        )
    return frame, len(feed)


def request_news(key, task):
    ticker, start, end = task
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": start.strftime("%Y%m%dT0000"),
        "time_to": (end + timedelta(days=1)).strftime("%Y%m%dT0000"),
        "sort": "EARLIEST",
        "limit": LIMIT,
        "apikey": key,
    }
    try:
        with urlopen(URL + "?" + urlencode(params), timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        raise ValueError(
            f"Provider HTTP status {error.code}; no files marked successful."
        ) from None
    except (URLError, TimeoutError, json.JSONDecodeError):
        raise ValueError(
            "Network/JSON response error; request may have used API quota. Retry later."
        ) from None


def collect(
    root,
    tasks,
    state,
    audit,
    key,
    max_requests=3,
    daily_budget=25,
    delay=15,
    fetch=request_news,
    clock=time.time,
    sleeper=time.sleep,
):
    count = 0
    state_path = audit / "collection_status.json"
    state["last_stop"] = None
    while tasks:
        if count >= max_requests:
            state["last_stop"] = "Per-run request limit reached; rerun to continue."
            break
        now = clock()
        if sum(t > now - 86400 for t in state["request_times"]) >= daily_budget:
            state["last_stop"] = "Local rolling-24-hour request budget reached; wait for quota."
            break
        if state["request_times"]:
            wait = delay - (now - state["request_times"][-1])
            if wait > 0:
                sleeper(wait)
        task = tasks.pop(0)
        name = job_key(*task)
        state["request_times"].append(clock())
        count += 1
        record = {
            "ticker": task[0],
            "start": str(task[1]),
            "end": str(task[2]),
            "status": "requesting",
            "requested_utc": datetime.now(UTC).isoformat(),
            "coverage_verified": False,
        }
        state["records"][name] = record
        save_json(state_path, state)
        print(f"Request {count}/{max_requests}: {name}", flush=True)
        try:
            data = fetch(key, task)
            frame, feed_count = validate_response(data, task)
            safe_data = json.loads(json.dumps(data).replace(key, "[REDACTED]")) if key else data
            response_path = audit / "responses" / (name + ".json")
            save_json(response_path, safe_data)
            record.update(
                feed_count=feed_count,
                saved_article_count=len(frame),
                files={str(response_path.relative_to(root)): digest(response_path)},
            )
            if feed_count >= LIMIT:
                record["status"] = "capped"
                tasks[0:0] = children(*task)
                print("  Article cap reached; queued smaller intervals.", flush=True)
            else:
                record["status"] = "retrieved_below_cap" if len(frame) else "empty_unverified"
                if not frame.empty:
                    output = root / "data/raw/news" / task[0] / ("prd_" + name + ".parquet")
                    if output.exists():
                        pd.testing.assert_frame_equal(
                            pd.read_parquet(output).reset_index(drop=True),
                            frame.reset_index(drop=True),
                        )
                    else:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        temp = output.with_suffix(".parquet.tmp")
                        frame.to_parquet(temp, index=False)
                        temp.replace(output)
                    record["files"][str(output.relative_to(root))] = digest(output)
                print(
                    f"  {record['status']}: {len(frame)} articles. Coverage remains unverified.",
                    flush=True,
                )
        except (ValueError, TypeError, AssertionError, OSError) as error:
            message = str(error).replace(key, "[REDACTED]") if key else str(error)
            if record["status"] != "capped":
                record["status"] = "error"
            record["error"] = message
            state["last_stop"] = message
            save_json(state_path, state)
            print(f"Stopped: {message}", flush=True)
            return 1
        save_json(state_path, state)
    state["last_stop"] = (
        state["last_stop"]
        or "Planned retrieval finished. Coverage audit and processing still required."
    )
    state["requests_this_run"] = count
    save_json(state_path, state)
    print(state["last_stop"], flush=True)
    print(f"Status: {state_path}", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument(
        "--fetch", action="store_true", help="Make API requests; otherwise only show the plan."
    )
    parser.add_argument("--max-requests", type=int, default=3)
    parser.add_argument(
        "--daily-budget",
        type=int,
        default=25,
        help="Local rolling-24-hour ceiling; must fit your account quota.",
    )
    args = parser.parse_args()
    if not 1 <= args.max_requests <= args.daily_budget <= 25:
        parser.error("Use 1 <= max-requests <= daily-budget <= 25 for this conservative collector.")
    config_path = Path(args.config).resolve()
    root = config_path.parent
    config = yaml.safe_load(config_path.read_text())
    tickers = args.tickers or config["universe"]
    if len(set(tickers)) != len(tickers) or not set(tickers).issubset(config["universe"]):
        parser.error("Tickers must be unique members of the configured universe.")
    audit = root / "reports/prd/news_collection"
    audit.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (audit / "collection.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another collection process is active.")
        path = audit / "collection_status.json"
        scope = {k: config[k] for k in ("universe", "start_date")}
        scope["end_date"] = config["regression"]["end_date"]
        state = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "version": 1,
                "scope": scope,
                "records": {},
                "request_times": [],
                "coverage_policy": "Retrieval evidence only. No complete=True or neutral news fills. Legacy filenames are unverified.",
                "quota_note": "Only calls made by this collector are tracked; other scripts may already have used account quota.",
            }
        )
        if state["scope"] != scope:
            parser.error(
                "Collection scope changed; preserve the existing audit and investigate before continuing."
            )
        tasks, inventory = make_plan(root, config, tickers, state["records"], audit)
        state["legacy_inventory"] = inventory
        state["selected_tickers"] = tickers
        state["planned_request_intervals"] = len(tasks)
        save_json(path, state)
        print(
            f"Pending intervals for selected tickers: {len(tasks)} (capped intervals need additional requests)."
        )
        for task in tasks[:10]:
            print(" ", job_key(*task))
        print("Existing legacy files are preserved, not certified complete.")
        if not args.fetch or not tasks:
            return 0
        key = os.environ.get("ALPHAVANTAGE_API_KEY") or getpass.getpass(
            "Alpha Vantage API key (hidden): "
        )
        if not key.strip():
            parser.error("An API key is required. Enter it locally, never upload it.")
        return collect(root, tasks, state, audit, key.strip(), args.max_requests, args.daily_budget)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped. Completed files are saved; the interrupted request may have used quota.")
        raise SystemExit(130) from None
    except (ValueError, OSError) as error:
        print(f"Collection stopped: {error}")
        raise SystemExit(1) from None
