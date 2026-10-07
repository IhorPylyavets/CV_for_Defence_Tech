#!/usr/bin/env python3
"""Serve the WebGL viewer (stdlib only): python serve.py [--port 8000] [--no-browser]"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import threading
import webbrowser
from typing import Any

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer")


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map: dict[str, str] = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                                      ".js": "text/javascript", ".f16": "application/octet-stream"}

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    httpd = http.server.ThreadingHTTPServer((args.host, args.port), functools.partial(Handler, directory=ROOT))
    url = f"http://{args.host}:{args.port}/"
    print(f"NeRF Voxel Explorer at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
