"""Paths, credentials, and HTTP defaults.

Credentials come from the environment (or a repo-root .env, loaded if present):
  ALSI_KEY      — free key from https://alsi.gie.eu/account (x-key header)
  AISSTREAM_KEY — free key from https://aisstream.io (weeks 3-4)
  ENTSOE_TOKEN  — free token from ENTSO-E transparency (weeks 7-8, event study)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DATA_DIR = Path(os.environ.get("LNG_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
REFERENCE_DIR = DATA_DIR / "reference"
REPORTS_DIR = REPO_ROOT / "reports"

ALSI_KEY = os.environ.get("ALSI_KEY", "").strip()
AISSTREAM_KEY = os.environ.get("AISSTREAM_KEY", "").strip()
ENTSOE_TOKEN = os.environ.get("ENTSOE_TOKEN", "").strip()

# Identify ourselves politely to the public APIs.
USER_AGENT = "lng-nowcast/0.1 (student research; github repo forthcoming)"
HTTP_TIMEOUT = 90  # seconds
RETRIES = 4
RETRY_BACKOFF = 3.0  # seconds, doubled per attempt


def ensure_dirs() -> None:
    for d in (RAW_DIR, SNAPSHOT_DIR, REFERENCE_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
