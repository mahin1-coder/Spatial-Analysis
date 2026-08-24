#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


DRIVE_IDS = {
    "TOR7_BEST_AFTER_win45d.tif": "1Wu1j3l8MljL4nCbC0gPMoz-wvhIUK15a",
    "TOR10_BEST_BEFORE_win25d.tif": "1kW0walkfyuVprPJ9erkIo1_RlyQcdNSl",
    "TOR11_BEST_BEFORE_win35d.tif": "1jvprx7P0itFcrxnEw_iirPYDLR3wQTNp",
    "TOR101_BEST_BEFORE_win45d.tif": "1LUecikLs3daxXH7yDDZxhlw4q8QPlkJj",
    "TOR102_BEST_AFTER_win45d.tif": "1CMlpMXsCrIoTgGwdiEGrIv2YpzGkKOVt",
    "TOR102_BEST_BEFORE_win10d.tif": "1GX3nJdapeIvfZL8Q_o2K1Su-bMr6W1d-",
    "TOR105_BEST_AFTER_win10d.tif": "1QfZ4n59GmATA3htHfKJyR9zntaacT1tz",
    "TOR111_BEST_AFTER_win50d.tif": "1w69PH-7658s0Ydq2RJZo9eGPpr9MSEbm",
    "TOR112_BEST_AFTER_win50d.tif": "1-Kf-celQQbbFVvH969UxfKjSz3MHWm8P",
    "TOR112_BEST_BEFORE_win45d.tif": "184JKtip-7hMh166spPWQW45JY8KnWZ5d",
    "TOR114_BEST_BEFORE_win55d.tif": "19_DuSVJfHN6RKPNAKYH4x1xAUvxyRcCn",
    "TOR115_BEST_AFTER_win10d.tif": "1ajPMgZKWuRjCKW_jzqyVAPUAZF6QbQMj",
    "TOR123_BEST_BEFORE_win15d.tif": "1xWRcZLxtUuxbgh68kVN9Y0MQ1tHxBTM1",
    "TOR123_BEST_AFTER_win10d.tif": "1PYLivaOIeCNqmhSxh_8hxVSyLv9Ik3Fx",
    "TOR69_BEST_AFTER_win25d.tif": "1FMAplOWWdptsxA-yFpmWCVrUgmx5QkL8",
    "TOR69_BEST_BEFORE_win45d.tif": "1jfXY5sTlcbKG6Vc8UM-a8_bmXzH6_k6Z",
    "TOR70_BEST_AFTER_win40d.tif": "1LdgK3SIqnvrNdfJNdrDUtdLzbnmz4pLb",
    "TOR77_BEST_BEFORE_win25d.tif": "1VXUYBsRaE7IgN6yCqPaYV0a2Bzr15cXL",
    "TOR90_BEST_AFTER_win25d.tif": "1gRBV5wCeU7zyKPJcgFa3rcqpAetnwrBC",
    "TOR95_BEST_AFTER_win35d.tif": "1I738GtMnk52qSle2yYV5o5r8cauiuy0G",
    "TOR12_BEST_BEFORE_win25d.tif": "16_hW5fdpaEOBBlMTLT8OkbyXzMAN63pc",
    "TOR13_BEST_AFTER_win10d.tif": "1R3H7tgs31kZW7UW3KFL3kFJXBUcjFnS0",
    "TOR16_BEST_AFTER_win15d.tif": "1oDPVjKftx0OGt5DR13fynue7Y36bK3vH",
    "TOR18_BEST_BEFORE_win20d.tif": "10nkIFRnuI9VoO55Yz17aq7nRMjaAutrd",
    "TOR5_BEST_BEFORE_win20d.tif": "1SQl556QvdC1Z6XXHnztKPNNzXYVPFHQw",
    "TOR61_BEST_AFTER_win40d.tif": "1l85A4R65kOPxaPiwCvX5o6b1MD1UKp9K",
    "TOR62_BEST_AFTER_win35d.tif": "1qeBspDB1EuRsih4AWQ72VbVkRKUYpzhX",
    "TOR62_BEST_BEFORE_win35d.tif": "1hz9uJW4k0pcGlCQzXLRWiSu4vpXslxpP",
    "TOR66_BEST_BEFORE_win35d.tif": "1U_ftM4xZs4h5LGcWO6j_HCvedlXQA2h1",
    "TOR68_BEST_AFTER_win30d.tif": "15hpqtECt1e34XF4pztDqGoP8r6Mr4KdV",
    "TOR68_BEST_BEFORE_win55d.tif": "1UfNAF4tEggwLiBJ5Ej5w2bZ6laGGH1oo",
}


def is_materialized(path: Path) -> bool:
    return path.exists() and path.stat().st_blocks > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore dataless TIFF placeholders from Google Drive.")
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()
    paths = {path.name: path for root in args.roots for path in root.glob("*.tif")}
    restored = 0
    for name, drive_id in DRIVE_IDS.items():
        target = paths.get(name)
        if target is None or is_materialized(target):
            continue
        temporary = target.with_suffix(".download")
        url = f"https://drive.usercontent.google.com/download?id={drive_id}&export=download&confirm=t"
        print(f"[restore] {name}", flush=True)
        subprocess.run(["curl", "-fL", "--retry", "4", "--retry-delay", "2", url, "-o", str(temporary)], check=True)
        with temporary.open("rb") as stream:
            header = stream.read(4)
        if header not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded bytes for {name} are not a TIFF.")
        os.replace(temporary, target)
        restored += 1
    remaining = [str(path) for path in paths.values() if not is_materialized(path)]
    print(f"[restore] restored={restored} remaining={len(remaining)}")
    if remaining:
        print("\n".join(remaining))
    return 1 if remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
