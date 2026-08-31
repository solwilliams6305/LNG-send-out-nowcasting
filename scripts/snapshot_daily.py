#!/usr/bin/env python3
"""Twice-daily snapshot entrypoint (GitHub Actions cron / local run).

Scheduled at 18:45 and 22:15 UTC so that, winter or summer, runs land after
ALSI's 19:30 CET first publication and its 23:00 CET late-reporter pass.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lng_nowcast import snapshot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-days", type=int, default=45,
                    help="trailing gas days to re-fetch each run (default 45)")
    ap.add_argument("--skip-hourly", action="store_true")
    ap.add_argument("--ais-window", type=float, default=480,
                    help="AIS listening window in seconds (default 480; 0 disables)")
    args = ap.parse_args()

    m = snapshot.run(window_days=args.window_days,
                     include_hourly=not args.skip_hourly,
                     ais_window_s=args.ais_window)
    print(
        f"snapshot {m['snapshot_utc']}: entsog={m['entsog_rows']} rows, "
        f"hourly={m['entsog_hourly_rows']}, alsi={m['alsi_rows']}, "
        f"nationalgas={m['nationalgas_rows']}, ais={m['ais_rows']} "
        f"(keys: alsi={m['alsi_key_present']}, ais={m['ais_key_present']})"
    )
    if m["errors"]:
        print(f"errors: {m['errors']}", file=sys.stderr)
    # Fail the job only if nothing at all was captured.
    total = (m["entsog_rows"] + m["entsog_hourly_rows"] + m["alsi_rows"]
             + m["nationalgas_rows"])
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
