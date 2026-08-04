#!/usr/bin/env python
"""Local control panel: pick a Z-stack folder, run the analysis, open the viewer.

    ./.venv/bin/python serve.py

Then open http://localhost:8765.

Standard library only -- no Flask, no extra dependency. It runs the pipeline in a
worker thread and streams its log to the page, so the whole analysis can be
driven from the browser you are presenting from.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import threading
import time
import traceback
import webbrowser
from collections import deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

_Z_RE = re.compile(r"_Z(\d+)_")


# --------------------------------------------------------------------------- #
# job state
# --------------------------------------------------------------------------- #

class Job:
    """One pipeline run, with its log captured for the browser."""

    def __init__(self):
        self.lock = threading.Lock()
        self.lines: deque[str] = deque(maxlen=4000)
        self.seq = 0
        self.running = False
        self.folder = ""
        self.error: str | None = None
        self.finished_at: float | None = None
        self.result_dir: str | None = None

    # The pipeline writes progress with carriage returns; treat "\r" as
    # "rewrite the current line" so the browser sees a progress bar rather than
    # thousands of near-identical lines.
    def write(self, text: str) -> None:
        with self.lock:
            for chunk in text.replace("\r\n", "\n").split("\n"):
                if "\r" in chunk:
                    chunk = chunk.split("\r")[-1]
                    if self.lines:
                        self.lines[-1] = chunk
                    else:
                        self.lines.append(chunk)
                elif chunk or text.endswith("\n"):
                    self.lines.append(chunk)
            self.seq += 1

    def flush(self) -> None:
        pass

    def snapshot(self, since: int) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "folder": self.folder,
                "error": self.error,
                "seq": self.seq,
                "result_dir": self.result_dir,
                "lines": list(self.lines) if since != self.seq else None,
            }


JOB = Job()


def _run_job(folder: Path, outdir: Path, opts: dict) -> None:
    import contextlib

    from jx3d.config import Params
    from jx3d.pipeline import run

    JOB.running = True
    JOB.error = None
    JOB.result_dir = None
    JOB.folder = str(folder)
    with JOB.lock:
        JOB.lines.clear()

    try:
        params = Params(
            mode=opts.get("mode", "both"),
            detector=opts.get("detector", "cellpose"),
            z_step=max(1, int(opts.get("z_step", 1))),
            expected_diameter_px=float(opts.get("diameter", 40)),
            min_diameter_px=float(opts.get("min_diameter", 8)),
            max_diameter_px=float(opts.get("max_diameter", 160)),
            fit_dome=bool(opts.get("fit_dome", True)),
        )
        cal = {}
        if opts.get("px_size"):
            cal["px_um"] = float(opts["px_size"])
        if opts.get("z_step_um"):
            cal["z_um"] = float(opts["z_step_um"])

        with contextlib.redirect_stdout(JOB), contextlib.redirect_stderr(JOB):
            res = run(folder, outdir, params=params,
                      gpu=not opts.get("no_gpu", False),
                      use_cache=not opts.get("no_cache", False),
                      min_sharpness=float(opts.get("min_sharpness", 0.25)),
                      calibration=cal)
        JOB.result_dir = str(res.outdir)
    except Exception:
        JOB.error = traceback.format_exc(limit=6)
        JOB.write("\n!! FAILED\n" + JOB.error)
    finally:
        JOB.running = False
        JOB.finished_at = time.time()
        JOB.seq += 1


# --------------------------------------------------------------------------- #
# filesystem scanning
# --------------------------------------------------------------------------- #

def _stack_info(d: Path) -> dict | None:
    tifs = [p for p in d.glob("*.tif") if _Z_RE.search(p.name)]
    if len(tifs) < 3:
        return None
    gci = list(d.glob("*.gci")) + list(d.parent.glob("*.gci"))
    out = Path("output") / d.name
    info = {
        "path": str(d.resolve()),
        "name": d.name,
        "slices": len(tifs),
        "calibrated": bool(gci),
        "done": (out / "organoids.json").exists(),
        "organoids": None,
    }
    if info["done"]:
        try:
            info["organoids"] = json.loads(
                (out / "organoids.json").read_text())["n_organoids"]
        except Exception:
            pass
    return info


def scan(root: Path) -> dict:
    root = root.resolve()
    stacks, dirs = [], []
    try:
        entries = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError as exc:
        return {"root": str(root), "error": str(exc), "stacks": [], "dirs": []}

    here = _stack_info(root)
    if here:
        stacks.append(here)
    for d in entries:
        info = _stack_info(d)
        if info:
            stacks.append(info)
        else:
            dirs.append({"path": str(d.resolve()), "name": d.name})
    return {
        "root": str(root),
        "parent": str(root.parent) if root.parent != root else None,
        "stacks": stacks,
        "dirs": dirs,
    }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

PAGE = (Path(__file__).parent / "jx3d" / "panel.html")


class Handler(BaseHTTPRequestHandler):
    server_version = "JX3D"

    def log_message(self, *a):        # keep the console for the pipeline only
        pass

    # ------------------------------------------------------------------ utils
    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj).encode("utf-8"), "application/json", code)

    def _file(self, path: Path) -> None:
        if not path.is_file():
            self._json({"error": f"not found: {path}"}, 404)
            return
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".png": "image/png", ".jpg": "image/jpeg",
            ".json": "application/json", ".csv": "text/csv",
            ".ply": "application/octet-stream",
        }.get(path.suffix, "application/octet-stream")
        self._send(path.read_bytes(), ctype)

    # -------------------------------------------------------------------- GET
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = unquote(u.path)

        if p in ("/", "/index.html"):
            self._file(PAGE)
        elif p == "/api/scan":
            self._json(scan(Path(q.get("root", [str(Path.cwd())])[0])))
        elif p == "/api/status":
            self._json(JOB.snapshot(int(q.get("since", ["-1"])[0])))
        elif p == "/api/results":
            out = Path("output")
            res = []
            if out.is_dir():
                for d in sorted(out.iterdir()):
                    j = d / "organoids.json"
                    if not j.is_file():
                        continue
                    try:
                        meta = json.loads(j.read_text())
                    except Exception:
                        continue
                    from jx3d.viewer import template_version
                    res.append({
                        "name": d.name,
                        "organoids": meta.get("n_organoids"),
                        "calibrated": meta.get("units", {}).get("calibrated"),
                        "dome": bool(meta.get("dome")),
                        "viewer": (d / "viewer.html").exists(),
                        # A viewer is a frozen copy of the template; if the
                        # template has moved on, this page still carries
                        # whatever was wrong with it when it was written.
                        "stale": meta.get("viewer_version") != template_version(),
                    })
            self._json(res)
        elif p.startswith("/out/"):
            self._file(Path("output") / p[len("/out/"):])
        else:
            self._json({"error": "no such endpoint"}, 404)

    # ------------------------------------------------------------------- POST
    def do_POST(self):
        p = unquote(urlparse(self.path).path)
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "bad json"}, 400)
            return

        if p == "/api/run":
            if JOB.running:
                self._json({"error": "a run is already in progress"}, 409)
                return
            folder = Path(body.get("folder", ""))
            if not folder.is_dir():
                self._json({"error": f"no such folder: {folder}"}, 400)
                return
            outdir = Path("output") / folder.name
            threading.Thread(target=_run_job, args=(folder, outdir, body),
                             daemon=True).start()
            self._json({"started": True, "outdir": str(outdir)})
        else:
            self._json({"error": "no such endpoint"}, 404)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="JX-3D control panel")
    ap.add_argument("--root", default=".", help="folder to start browsing from")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args(argv)

    url = f"http://localhost:{a.port}/?root={Path(a.root).resolve()}"
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"JX-3D control panel: {url}")
    print("Ctrl-C to stop.")
    if not a.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
