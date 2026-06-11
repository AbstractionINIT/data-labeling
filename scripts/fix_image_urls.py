"""
Make task image URLs RELATIVE so they load over both localhost and the LAN.

Label Studio's local-files sync can bake an absolute host (e.g.
http://localhost:8090/...) and Windows backslashes into each task's image URL.
That breaks images when you open the UI from another device (the <img> points
at localhost, cross-origin, so your session cookie isn't sent -> 401).

This rewrites every task's `data.image` to a host-less, forward-slash form:
    /data/local-files/?d=images/<file>
which the browser resolves against whatever origin you're viewing on.

Run:  python scripts/fix_image_urls.py
"""
from __future__ import annotations

import os
import sys
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests

LS = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8090").rstrip("/")
TOK = os.environ.get("LABEL_STUDIO_API_KEY", "")
TITLE = os.environ.get("PROJECT_TITLE", "Construction Site Detection")
H = {"Authorization": f"Token {TOK}", "Content-Type": "application/json"}


def to_relative(url: str) -> str | None:
    """http://host/data/local-files/?d=images\\x.jpg -> /data/local-files/?d=images/x.jpg"""
    if not url:
        return None
    q = parse_qs(urlparse(url).query)
    d = unquote((q.get("d") or [""])[0]).replace("\\", "/")
    if not d:
        return None
    return "/data/local-files/?d=" + quote(d, safe="/")


def main():
    if not TOK:
        sys.exit("LABEL_STUDIO_API_KEY not set (source scripts/env.ps1 / env.sh first).")
    r = requests.get(f"{LS}/api/projects", headers=H, timeout=30)
    r.raise_for_status()
    items = r.json().get("results", r.json() if isinstance(r.json(), list) else [])
    proj = next((p for p in items if p.get("title") == TITLE), items[0] if items else None)
    if not proj:
        sys.exit(f"Project '{TITLE}' not found.")
    pid = proj["id"]

    changed = skipped = 0
    page = 1
    while True:
        tr = requests.get(f"{LS}/api/tasks", headers=H,
                          params={"project": pid, "page": page, "page_size": 200}, timeout=60)
        if tr.status_code == 404:
            break
        tr.raise_for_status()
        body = tr.json()
        tasks = body.get("tasks", body if isinstance(body, list) else [])
        if not tasks:
            break
        for t in tasks:
            cur = (t.get("data") or {}).get("image")
            rel = to_relative(cur or "")
            if rel and rel != cur:
                data = dict(t.get("data") or {}); data["image"] = rel
                requests.patch(f"{LS}/api/tasks/{t['id']}", headers=H,
                               json={"data": data}, timeout=30).raise_for_status()
                changed += 1
            else:
                skipped += 1
        page += 1

    print(f"Done. Rewrote {changed} task image URLs to relative form "
          f"({skipped} already fine). Hard-refresh Label Studio (Ctrl+F5).")


if __name__ == "__main__":
    main()
