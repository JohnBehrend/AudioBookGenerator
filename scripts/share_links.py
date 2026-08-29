#!/usr/bin/env python3
"""Print current shareable, authenticated download links for generated audiobooks.

Detects the machine's current public IP (and LAN IP) and prints opencode
/file/download links with the auth_token embedded, so they can be copied and
shared. Handles dynamic public IPs automatically.
"""

import base64
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

PORT = 8080
USERNAME = "opencode"

PASSWORD_ENV = "OPENCODE_SERVER_PASSWORD"
DEFAULT_PASSWORD = "piggypants"


def public_ip() -> str:
    for service in ("https://api.ipify.org", "https://ifconfig.me", "https://icanhazip.com"):
        try:
            out = subprocess.run(
                ["curl", "-s", "--max-time", "8", service],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            if out:
                return out
        except Exception:
            continue
    return ""


def lan_ip() -> str:
    try:
        out = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return out.split()[0] if out else ""
    except Exception:
        return ""


def auth_token() -> str:
    password = os.environ.get(PASSWORD_ENV, DEFAULT_PASSWORD)
    return base64.b64encode(f"{USERNAME}:{password}".encode()).decode()


def make_link(host: str, path: str) -> str:
    token = auth_token()
    return (
        f"http://{host}:{PORT}/file/download"
        f"?path={quote(path)}&auth_token={quote(token)}"
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    base = root / "voice_test" / "tbi_output"
    files = sorted(
        str(p) for p in base.glob("*.m4b")
    ) if base.is_dir() else []

    if not files:
        print("No .m4b files found under voice_test/*_output/.", file=sys.stderr)
        return

    pub = public_ip()
    lan = lan_ip()

    print("Shareable download links (auth embedded):\n")
    for f in files:
        rel = str(Path(f).relative_to(Path("/home/johnbehrend")))
        name = Path(f).name
        print(f"[{name}]")
        if pub:
            print(f"  Public : {make_link(pub, rel)}")
        if lan:
            print(f"  LAN    : {make_link(lan, rel)}")
        print()

    if not pub:
        print("(Could not determine public IP.)", file=sys.stderr)


if __name__ == "__main__":
    main()
