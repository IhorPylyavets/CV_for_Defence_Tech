#!/usr/bin/env python3
"""
Backend for the COLMAP sparse-reconstruction explorer: python server.py [--port 8002] [--no-browser]

  /                       viewer/ (static WebGL2 app)
  /data/<path>            raw files from data/ (images)
  GET  /api/datasets      datasets in data/ and their models (reference sparse/, runs/*/sparse/*, snapshots)
  GET  /api/model?path=   a model as JSON: cameras, image poses, points (xyz, rgb, error, track length)
  GET  /api/image?path=&id=   2D keypoints of one image and the 3D point each one triangulates
  GET  /api/point?path=&id=   track of one 3D point: (image id, 2D position) for every observation
  GET  /api/thumb?dataset=&name=&size=   downscaled JPEG (needs Pillow, otherwise the original is sent)
  POST /api/run           start reconstruct.py with the posted options (one job at a time)
  POST /api/download      start download_data.py
  POST /api/cancel        stop the running job
  POST /api/delete        delete a run folder
  GET  /api/status?since= job state, parsed progress, log lines from index `since`
"""
from __future__ import annotations

import argparse
import functools
import gzip
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from collections import OrderedDict
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.join(HERE, "viewer")
DATA_DIR = os.path.realpath(os.path.join(HERE, "data"))

try:
    import pycolmap
except ImportError:  # the viewer still starts, and shows this message when a model is opened
    pycolmap = None

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None


def safe_path(rel: str, root: str = DATA_DIR) -> str:
    """Resolve rel inside root, refusing anything that escapes it."""
    p = os.path.realpath(os.path.join(root, rel))
    if p != root and not p.startswith(root + os.sep):
        raise ValueError("path outside data/")
    return p


def is_model_dir(d: str) -> bool:
    return any(os.path.exists(os.path.join(d, f"cameras.{e}")) for e in ("bin", "txt"))


# --------------------------------------------------------------------------- #
# Datasets and models on disk
# --------------------------------------------------------------------------- #
def list_datasets() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not os.path.isdir(DATA_DIR):
        return out
    for name in sorted(os.listdir(DATA_DIR)):
        ds = os.path.join(DATA_DIR, name)
        img_dir = os.path.join(ds, "images")
        if not os.path.isdir(img_dir):
            continue
        models: list[dict[str, Any]] = []
        sparse = os.path.join(ds, "sparse")
        if is_model_dir(sparse):
            models.append({"path": f"{name}/sparse", "label": "reference (shipped with the dataset)", "kind": "reference"})
        elif os.path.isdir(sparse):
            for k in sorted(os.listdir(sparse)):
                if is_model_dir(os.path.join(sparse, k)):
                    models.append({"path": f"{name}/sparse/{k}", "label": f"reference {k}", "kind": "reference"})
        runs_dir = os.path.join(ds, "runs")
        for run in sorted(os.listdir(runs_dir), reverse=True) if os.path.isdir(runs_dir) else []:
            rd = os.path.join(runs_dir, run)
            params: dict[str, Any] = {}
            try:
                with open(os.path.join(rd, "params.json")) as f:
                    params = json.load(f)
            except (OSError, ValueError):
                pass
            snaps_dir = os.path.join(rd, "snapshots")
            snaps = sorted(s for s in os.listdir(snaps_dir) if is_model_dir(os.path.join(snaps_dir, s))) \
                if os.path.isdir(snaps_dir) else []
            snaps = [f"{name}/runs/{run}/snapshots/{s}" for s in snaps]
            sp = os.path.join(rd, "sparse")
            ks = sorted(k for k in os.listdir(sp) if is_model_dir(os.path.join(sp, k))) if os.path.isdir(sp) else []
            for k in ks:
                models.append({"path": f"{name}/runs/{run}/sparse/{k}", "label": f"run {run}" + (f" / model {k}" if len(ks) > 1 else ""),
                               "kind": "run", "run": f"{name}/runs/{run}", "params": run_summary(params), "snapshots": snaps})
            if not ks:
                models.append({"path": snaps[-1] if snaps else None, "label": f"run {run} (no final model)",
                               "kind": "run", "run": f"{name}/runs/{run}", "params": run_summary(params),
                               "snapshots": snaps, "incomplete": True})
        n_img = len([f for f in os.listdir(img_dir) if not f.startswith(".")])
        out.append({"name": name, "num_images": n_img, "has_database": os.path.exists(os.path.join(ds, "database.db")),
                    "models": models})
    return out


def run_summary(p: dict[str, Any]) -> str:
    if not p:
        return ""
    n = len(p.get("images", []))
    if p.get("reuse_db"):
        feat = "reused database.db"
    else:
        feat = f"{p.get('camera_model')}, {p.get('max_image_size')}px, {p.get('matcher')}"
    return f"{n} images · {feat} · {p.get('mapper')}"


# --------------------------------------------------------------------------- #
# Reconstruction -> JSON
# --------------------------------------------------------------------------- #
class ModelCache:
    def __init__(self, size: int = 8) -> None:
        self.size = size
        self.items: OrderedDict[str, tuple[float, pycolmap.Reconstruction]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, rel: str) -> pycolmap.Reconstruction:
        if pycolmap is None:
            raise RuntimeError("pycolmap is not installed: pip install -r ../requirements.txt")
        path = safe_path(rel)
        if not is_model_dir(path):
            raise FileNotFoundError(rel)
        mtime = max(os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path))
        with self.lock:
            hit = self.items.get(rel)
            if hit and hit[0] == mtime:
                self.items.move_to_end(rel)
                return hit[1]
        rec = pycolmap.Reconstruction(path)
        with self.lock:
            self.items[rel] = (mtime, rec)
            while len(self.items) > self.size:
                self.items.popitem(last=False)
        return rec


MODELS = ModelCache()


def val(x: Any) -> Any:
    """pycolmap exposes some counters as methods and some as properties, depending on the version."""
    return x() if callable(x) else x


def pose_of(image: pycolmap.Image) -> tuple[np.ndarray, np.ndarray]:
    cfw = image.cam_from_world
    if callable(cfw):  # pycolmap >= 3.12 (rigs/frames): method instead of property
        cfw = cfw()
    return cfw.rotation.matrix(), cfw.translation


def model_json(rel: str) -> dict[str, Any]:
    rec = MODELS.get(rel)
    cams: dict[int, dict[str, Any]] = {}
    for cid, c in rec.cameras.items():
        cams[cid] = {"id": cid, "model": c.model_name, "width": c.width, "height": c.height,
                     "fx": c.focal_length_x, "fy": c.focal_length_y,
                     "cx": c.principal_point_x, "cy": c.principal_point_y,
                     "params": [round(float(v), 6) for v in c.params]}
    images: list[dict[str, Any]] = []
    for iid in sorted(rec.reg_image_ids()):
        im = rec.image(iid)
        R, t = pose_of(im)
        images.append({"id": iid, "name": im.name, "camera_id": im.camera_id,
                       "R": [round(float(v), 7) for v in R.reshape(-1)],
                       "t": [round(float(v), 6) for v in t],
                       "num_points2D": val(im.num_points2D), "num_points3D": val(im.num_points3D)})
    n = rec.num_points3D()
    ids: list[int] = [0] * n
    xyz: list[float] = [0.0] * (3 * n)
    rgb: list[int] = [0] * (3 * n)
    err: list[float] = [0.0] * n
    track: list[int] = [0] * n
    for i, (pid, p) in enumerate(rec.points3D.items()):
        ids[i] = pid
        x = p.xyz
        xyz[3 * i:3 * i + 3] = (round(float(x[0]), 5), round(float(x[1]), 5), round(float(x[2]), 5))
        c = p.color
        rgb[3 * i:3 * i + 3] = (int(c[0]), int(c[1]), int(c[2]))
        err[i] = round(float(p.error), 3)
        track[i] = p.track.length()
    ds = rel.split("/")[0]
    return {"path": rel, "dataset": ds, "cameras": list(cams.values()), "images": images,
            "points": {"id": ids, "xyz": xyz, "rgb": rgb, "err": err, "track": track},
            "stats": {"num_images": rec.num_images(), "num_reg_images": rec.num_reg_images(),
                      "num_points3D": n, "num_observations": rec.compute_num_observations(),
                      "mean_track_length": rec.compute_mean_track_length(),
                      "mean_observations_per_image": rec.compute_mean_observations_per_reg_image(),
                      "mean_reprojection_error": rec.compute_mean_reprojection_error()}}


def image_json(rel: str, iid: int) -> dict[str, Any]:
    rec = MODELS.get(rel)
    im = rec.image(iid)
    xy: list[float] = []
    p3d: list[int] = []
    for p in im.points2D:
        xy += [round(float(p.xy[0]), 2), round(float(p.xy[1]), 2)]
        p3d.append(int(p.point3D_id) if p.has_point3D() else -1)
    return {"id": iid, "name": im.name, "xy": xy, "point3D": p3d}


def point_json(rel: str, pid: int) -> dict[str, Any]:
    rec = MODELS.get(rel)
    p = rec.point3D(pid)
    obs: list[dict[str, Any]] = []
    for e in p.track.elements:
        xy = rec.image(e.image_id).points2D[e.point2D_idx].xy
        obs.append({"image_id": e.image_id, "point2D_idx": e.point2D_idx, "xy": [float(xy[0]), float(xy[1])]})
    return {"id": pid, "xyz": [float(v) for v in p.xyz], "rgb": [int(v) for v in p.color],
            "reproj_error": float(p.error), "track": obs}


def thumbnail(dataset: str, name: str, size: int | str) -> tuple[str, str]:
    src = safe_path(os.path.join(dataset, "images", name))
    if PILImage is None:
        return src, "image/jpeg"
    px = max(64, min(int(size), 4096))
    cache = safe_path(os.path.join(dataset, ".thumbs", str(px), os.path.splitext(name)[0] + ".jpg"))
    if not os.path.exists(cache) or os.path.getmtime(cache) < os.path.getmtime(src):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with PILImage.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((px, px), PILImage.LANCZOS)
            im.save(cache + ".part", "JPEG", quality=88)
        os.replace(cache + ".part", cache)
    return cache, "image/jpeg"


# --------------------------------------------------------------------------- #
# Background job (reconstruct.py / download_data.py in a subprocess)
# --------------------------------------------------------------------------- #
RE_FEATURE = re.compile(r"Processed file \[(\d+)/(\d+)\]")
RE_SEQ = re.compile(r"Processing image \[(\d+)/(\d+)\]")
RE_BLOCK = re.compile(r"Processing block \[(\d+)/(\d+), (\d+)/(\d+)\]")
RE_DOWNLOAD = re.compile(r"\(\s*([\d.]+) %\)")
RE_GLOG = re.compile(r"^[IWEF]\d{8} [\d:.]+\s+\d+ ")

# whitelisted reconstruct.py options: name -> (flag, type)
RUN_OPTS: dict[str, tuple[str, type]] = {
    "dataset": ("--dataset", str), "run": ("--run", str), "first": ("--first", int), "count": ("--count", int),
    "step": ("--step", int), "reuse_db": ("--reuse-db", bool), "camera_model": ("--camera-model", str),
    "camera_mode": ("--camera-mode", str), "max_image_size": ("--max-image-size", int),
    "max_features": ("--max-features", int), "matcher": ("--matcher", str), "overlap": ("--overlap", int),
    "guided_matching": ("--guided-matching", bool), "mapper": ("--mapper", str),
    "min_num_matches": ("--min-num-matches", int), "no_refine_focal": ("--no-refine-focal", bool),
    "no_refine_extra": ("--no-refine-extra", bool), "snapshot_every": ("--snapshot-every", int),
    "num_threads": ("--num-threads", int), "seed": ("--seed", int),
}


class Job:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[str] | None = None
        self.reset("idle")

    def reset(self, kind: str) -> None:
        self.kind: str = kind
        self.state: str = "idle" if kind == "idle" else "running"
        self.returncode: int | None = None
        self.lines: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.run_dir: str | None = None
        self.models: list[dict[str, Any]] = []
        self.progress: dict[str, dict[str, Any]] = {}
        self.stage: str | None = None
        self.started: float = time.time()
        self.finished: float | None = None
        self.error: str | None = None
        self.log_file: IO[str] | None = None

    def start(self, kind: str, argv: list[str]) -> None:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                raise RuntimeError("a job is already running")
            self.reset(kind)
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "GLOG_logtostderr": "1"}
            self.proc = subprocess.Popen([sys.executable, "-u", *argv], cwd=HERE, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
            self.lines.append("$ python " + " ".join(os.path.relpath(a, HERE) if os.path.isabs(a) else a for a in argv))
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

    def _pump(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            with self.lock:
                if proc is not self.proc:
                    return
                self._parse(line)
                self.lines.append(RE_GLOG.sub("", line))
                if self.log_file:
                    self.log_file.write(line + "\n")
        rc = proc.wait()
        with self.lock:
            if proc is not self.proc:
                return
            self.returncode = rc
            self.finished = time.time()
            if self.state == "cancelled":
                pass
            elif rc == 0:
                self.state = "done"
            else:
                self.state = "failed"
                tail = [l for l in self.lines[-30:] if l.strip()]
                self.error = tail[-1] if tail else f"exit code {rc}"
            self.lines.append(f"[{self.state}] exit code {rc}, {self.finished - self.started:.1f} s")
            if self.log_file:
                self.log_file.close()
                self.log_file = None

    def _parse(self, line: str) -> None:
        pr = self.progress
        if line.startswith("@@ "):
            try:
                ev = json.loads(line[3:])
            except ValueError:
                return
            self.events.append(ev)
            st = ev.get("stage")
            if st == "start":
                self.run_dir = ev["run_dir"]
                try:
                    self.log_file = open(os.path.join(DATA_DIR, self.run_dir, "log.txt"), "w")
                    self.log_file.write("\n".join(self.lines) + "\n")
                except OSError:
                    pass
            elif st in ("features", "matching", "mapping"):
                p = pr.setdefault(st, {"done": 0, "total": ev.get("total", 0), "state": "running"})
                self.stage = st
                if ev.get("skipped"):
                    p.update(state="skipped")
                elif ev.get("done"):
                    p.update(state="failed" if ev.get("failed") else "done", seconds=ev.get("seconds"))
                    if not ev.get("failed") and st != "mapping":
                        p["done"] = p["total"]
                elif "registered" in ev:
                    p["done"] = min(ev["registered"], ev["total"])
            elif st == "database":
                pr["database"] = ev
            elif st == "finished":
                self.models = ev.get("models", [])
                self.stage = "finished"
            return
        if self.kind == "download":
            m = RE_DOWNLOAD.search(line)
            if m:
                pr["download"] = {"done": float(m.group(1)), "total": 100, "state": "running"}
            elif line.startswith("Extracting"):
                pr["download"] = {"done": 100, "total": 100, "state": "extracting"}
            return
        m = RE_FEATURE.search(line)
        if m and "features" in pr:
            pr["features"]["done"] = int(m.group(1))
            pr["features"]["total"] = int(m.group(2))
            return
        m = RE_SEQ.search(line)
        if m and "matching" in pr:
            pr["matching"]["done"], pr["matching"]["total"] = int(m.group(1)), int(m.group(2))
            return
        m = RE_BLOCK.search(line)
        if m and "matching" in pr:
            a, na, b, nb = map(int, m.groups())
            pr["matching"]["done"], pr["matching"]["total"] = (a - 1) * nb + b, na * nb
            pr["matching"]["unit"] = "blocks"

    def cancel(self) -> bool:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.state = "cancelled"
                self.proc.terminate()
                threading.Timer(3.0, lambda p=self.proc: p.poll() is None and p.kill()).start()
                return True
        return False

    def status(self, since: int) -> dict[str, Any]:
        with self.lock:
            snap = None
            if self.run_dir:
                sd = os.path.join(DATA_DIR, self.run_dir, "snapshots")
                if os.path.isdir(sd):
                    s = sorted(x for x in os.listdir(sd) if is_model_dir(os.path.join(sd, x)))
                    if s:
                        snap = f"{self.run_dir}/snapshots/{s[-1]}"
            since = max(0, min(since, len(self.lines)))
            return {"kind": self.kind, "state": self.state, "stage": self.stage, "returncode": self.returncode,
                    "error": self.error, "run_dir": self.run_dir, "progress": self.progress, "models": self.models,
                    "latest_snapshot": snap, "elapsed": (self.finished or time.time()) - self.started,
                    "log_start": since, "log": self.lines[since:since + 2000], "log_end": min(len(self.lines), since + 2000)}


JOB = Job()


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map: dict[str, str] = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript"}

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def send_json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, separators=(",", ":")).encode()
        gz = "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 4096
        if gz:
            body = gzip.compress(body, 5)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: str, ctype: str) -> None:
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self) -> None:
        url = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        try:
            if url.path == "/api/datasets":
                return self.send_json({"datasets": list_datasets(), "pycolmap": pycolmap and pycolmap.__version__,
                                       "pillow": PILImage is not None, "data_dir": DATA_DIR})
            if url.path == "/api/model":
                return self.send_json(model_json(q["path"]))
            if url.path == "/api/image":
                return self.send_json(image_json(q["path"], int(q["id"])))
            if url.path == "/api/point":
                return self.send_json(point_json(q["path"], int(q["id"])))
            if url.path == "/api/thumb":
                return self.send_file(*thumbnail(q["dataset"], q["name"], q.get("size", 1024)))
            if url.path == "/api/status":
                return self.send_json(JOB.status(int(q.get("since", 0))))
            if url.path.startswith("/data/"):
                p = safe_path(urllib.parse.unquote(url.path[len("/data/"):]))
                if not os.path.isfile(p):
                    return self.send_error(404)
                return self.send_file(p, self.guess_type(p))
        except FileNotFoundError as e:
            return self.send_json({"error": f"not found: {e}"}, 404)
        except (KeyError, ValueError) as e:
            return self.send_json({"error": f"bad request: {e}"}, 400)
        except Exception as e:  # noqa: BLE001 - report anything to the UI
            return self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)
        return super().do_GET()

    def do_POST(self) -> None:
        url = urllib.parse.urlparse(self.path)
        try:
            body = self.read_json()
            if url.path == "/api/run":
                if pycolmap is None:
                    raise RuntimeError("pycolmap is not installed: pip install -r ../requirements.txt")
                argv = [os.path.join(HERE, "reconstruct.py")]
                for k, v in body.items():
                    if k not in RUN_OPTS or v in (None, ""):
                        continue
                    flag, typ = RUN_OPTS[k]
                    if typ is bool:
                        if v:
                            argv.append(flag)
                    else:
                        val = str(typ(v))
                        if typ is str and not re.fullmatch(r"[\w.\-]+", val):
                            raise ValueError(f"invalid value for {k}")
                        argv += [flag, val]
                JOB.start("reconstruct", argv)
                return self.send_json({"ok": True})
            if url.path == "/api/download":
                JOB.start("download", [os.path.join(HERE, "download_data.py")])
                return self.send_json({"ok": True})
            if url.path == "/api/cancel":
                return self.send_json({"ok": JOB.cancel()})
            if url.path == "/api/delete":
                run = body["run"]
                parts = run.split("/")
                if len(parts) != 3 or parts[1] != "runs":
                    raise ValueError("can only delete data/<dataset>/runs/<run>")
                if JOB.state == "running" and JOB.run_dir == run:
                    raise RuntimeError("this run is still in progress")
                shutil.rmtree(safe_path(run))
                return self.send_json({"ok": True})
        except (KeyError, ValueError) as e:
            return self.send_json({"error": f"bad request: {e}"}, 400)
        except Exception as e:  # noqa: BLE001
            return self.send_json({"error": f"{type(e).__name__}: {e}"}, 409)
        self.send_error(404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    httpd = http.server.ThreadingHTTPServer((args.host, args.port), functools.partial(Handler, directory=VIEWER))
    url = f"http://{args.host}:{args.port}/"
    print(f"COLMAP Sparse Explorer at {url}  (Ctrl+C to stop)")
    if pycolmap is None:
        print("warning: pycolmap is not installed, models cannot be loaded or computed (pip install -r ../requirements.txt)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        JOB.cancel()


if __name__ == "__main__":
    main()
