"""Real-time run viewer for ``runs/*.jsonl`` training artifacts.

A tiny stdlib-only HTTP server. Lists all run artifacts under ``runs/``,
lets you pick any two from dropdowns (A / B), and overlays their metric
curves on the same chart. Polls the filesystem every two seconds so an
artifact being live-appended by an in-progress run updates in place — you
can watch a run's rel_L2 curve form as it trains.

The artifacts are JSONL: a ``meta`` header line, one ``train`` / ``val``
event line per log step (each with a nested ``metrics`` dict), and a
terminal ``end`` line. The reader tolerates a half-written final line, so
a live run is fine to view. Because metrics differ per experiment, the val
chart's metric is a dropdown (default ``rel_l2``), and both charts have a
log / linear y-axis toggle (log by default — the thing you actually want
for PINN error curves).

Usage:

    uv run -m experiments.viewer
    # then open http://localhost:8765 in a browser

    uv run -m experiments.viewer --port 9000 --runs-dir runs
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>gnome-optimizer run viewer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 14px 18px; color: #1a1a1a; }
  h2 { margin: 0 0 12px 0; font-size: 18px; }
  h3 { margin: 14px 0 6px 0; font-size: 13px; color: #555;
       text-transform: uppercase; letter-spacing: 0.04em; }
  .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
         margin-bottom: 8px; font-size: 13px; }
  .row label { display: flex; gap: 6px; align-items: center; }
  select { padding: 4px 6px; min-width: 360px; font-size: 12px;
           border: 1px solid #c0c0c0; border-radius: 4px; }
  select.metric { min-width: 120px; }
  .chart-wrap { width: 100%; max-width: 1200px; height: 300px;
                margin-bottom: 8px; }
  .summary { display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
             max-width: 1200px; margin-bottom: 12px; }
  .summary > div { background: #f8f8f8; padding: 8px 12px; border-radius: 4px;
                   font-size: 12px; }
  .summary code { font-family: ui-monospace, "Cascadia Code", Menlo, monospace; }
  table { font-size: 12px; border-collapse: collapse; max-width: 1200px;
          width: 100%; }
  td, th { border-bottom: 1px solid #eee; padding: 4px 8px; text-align: left;
           vertical-align: top; }
  th { background: #f4f4f4; }
  tr.diff td:nth-child(2), tr.diff td:nth-child(3) { background: #fff8d6; }
  .num { text-align: right; font-variant-numeric: tabular-nums;
         font-family: ui-monospace, "Cascadia Code", Menlo, monospace; }
  .a-color { color: #1f77b4; font-weight: 600; }
  .b-color { color: #d62728; font-weight: 600; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 10px;
          font-size: 11px; margin-left: 6px; }
  .ok { background: #d4edda; color: #155724; }
  .pending { background: #fff3cd; color: #856404; }
  #status { font-size: 11px; color: #777; }
</style>
</head>
<body>
<h2>gnome-optimizer run viewer</h2>
<div class="row">
  <label>since:
    <select id="since-filter">
      <option value="3600">last hour</option>
      <option value="21600">last 6 hours</option>
      <option value="86400" selected>last 24 hours</option>
      <option value="604800">last 7 days</option>
      <option value="2592000">last 30 days</option>
      <option value="0">all</option>
    </select>
  </label>
  <label>experiment:
    <select id="exp-filter" class="metric"><option value="">all</option></select>
  </label>
  <span id="filter-count"></span>
</div>
<div class="row">
  <label><span class="a-color">A</span>
    <select id="run-a"></select></label>
  <label><span class="b-color">B</span>
    <select id="run-b"></select></label>
  <label><input type="checkbox" id="auto-refresh" checked /> auto-refresh (2s)</label>
  <span id="status">—</span>
</div>
<div class="row">
  <label>val metric:
    <select id="metric" class="metric"></select></label>
  <label><input type="checkbox" id="logy" checked /> log y-axis</label>
  <label>train smoothing:
    <input type="number" id="smoothing-window" value="100" min="1" max="10000" step="10"
           style="width: 70px;" />
    <span style="color:#777; font-size:11px;">steps (1 = raw)</span>
  </label>
</div>

<div id="summary" class="summary"></div>

<h3 id="val-heading">Val metric</h3>
<div class="chart-wrap"><canvas id="val-chart"></canvas></div>

<h3>Train loss (per step, rolling mean <span id="train-window-label">100</span>-step)</h3>
<div class="chart-wrap"><canvas id="train-chart"></canvas></div>

<h3>Hyperparameters</h3>
<div id="meta"></div>

<script>
const COLOR_A = "#1f77b4";
const COLOR_B = "#d62728";

function makeChart(id, xLabel, yLabel) {
  const ctx = document.getElementById(id).getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: { datasets: [] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      parsing: false,
      scales: {
        x: { type: "linear", title: { display: true, text: xLabel } },
        y: { type: "logarithmic", title: { display: true, text: yLabel } },
      },
      elements: { point: { radius: 0 }, line: { borderWidth: 1.5, tension: 0 } },
      plugins: { legend: { display: true, position: "top", align: "end" } },
    },
  });
}

const charts = {
  val:   makeChart("val-chart",   "step", "rel_l2"),
  train: makeChart("train-chart", "step", "train loss"),
};

let runA = null, runB = null;
const ROLE = { A: { color: COLOR_A, label: "A" }, B: { color: COLOR_B, label: "B" } };

async function listRuns() {
  const r = await fetch("/api/runs");
  return r.ok ? await r.json() : [];
}
async function getRun(p) {
  const r = await fetch("/api/run?path=" + encodeURIComponent(p));
  return r.ok ? await r.json() : null;
}

// Build {x,y} points from a series object {steps:[], values:[]}, dropping any
// non-finite y (diverged runs log NaN) so the log scale doesn't choke.
function points(series) {
  if (!series) return [];
  const xs = series.steps, ys = series.values;
  const n = Math.min(xs.length, ys.length);
  const out = [];
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(ys[i])) out.push({ x: xs[i], y: ys[i] });
  }
  return out;
}

// Trailing (causal) rolling mean over a fixed step window — never "sees the
// future", so it's honest on a live curve.
function rollingMean(series, window) {
  const pts = points(series);
  if (!window || window <= 1 || pts.length === 0) return pts;
  const out = new Array(pts.length);
  let sum = 0;
  for (let i = 0; i < pts.length; i++) {
    sum += pts[i].y;
    if (i >= window) sum -= pts[i - window].y;
    out[i] = { x: pts[i].x, y: sum / Math.min(i + 1, window) };
  }
  return out;
}

// Chart.js is slow past a few thousand points; the per-step train curve can be
// 100k+. Stride it down for display (keep the last point).
function decimate(pts, maxPoints) {
  if (pts.length <= maxPoints) return pts;
  const stride = Math.ceil(pts.length / maxPoints);
  const out = pts.filter((_, i) => i % stride === 0);
  if (out[out.length - 1] !== pts[pts.length - 1]) out.push(pts[pts.length - 1]);
  return out;
}

function getSmoothingWindow() {
  const raw = parseInt(document.getElementById("smoothing-window").value, 10);
  return Number.isFinite(raw) && raw >= 1 ? raw : 1;
}

function dataset(role, pts) {
  return { label: ROLE[role].label, data: pts,
           borderColor: ROLE[role].color, backgroundColor: ROLE[role].color };
}

function setChart(chart, ptsA, ptsB) {
  const sets = [];
  if (ptsA && ptsA.length) sets.push(dataset("A", ptsA));
  if (ptsB && ptsB.length) sets.push(dataset("B", ptsB));
  chart.data.datasets = sets;
  chart.update("none");
}

function applyLogScale() {
  const type = document.getElementById("logy").checked ? "logarithmic" : "linear";
  charts.val.options.scales.y.type = type;
  charts.train.options.scales.y.type = type;
}

function getSinceCutoff() {
  const secs = parseInt(document.getElementById("since-filter").value, 10);
  return secs ? Date.now() / 1000 - secs : 0;
}

function filterRuns(runs) {
  const cutoff = getSinceCutoff();
  const exp = document.getElementById("exp-filter").value;
  // Always keep the currently-selected runs, even if they'd be filtered out —
  // narrowing a filter shouldn't silently un-select your pinned baseline.
  const pinned = new Set(["run-a", "run-b"]
    .map(id => document.getElementById(id).value).filter(Boolean));
  return runs.filter(r =>
    pinned.has(r.path) ||
    ((!cutoff || r.mtime >= cutoff) && (!exp || r.experiment === exp)));
}

function populateExpFilter(runs) {
  const sel = document.getElementById("exp-filter");
  const have = new Set([...sel.options].map(o => o.value));
  [...new Set(runs.map(r => r.experiment))].sort().forEach(e => {
    if (!have.has(e)) sel.add(new Option(e, e));
  });
}

function populateSelects(allRuns) {
  const filtered = filterRuns(allRuns);
  const html = ['<option value="">(none)</option>'];
  filtered.forEach(r => {
    const status = r.completed ? "" : " ⟳";
    html.push(`<option value="${r.path}">${r.experiment} / ${r.optimizer} / seed=${r.seed} / ${shortId(r.run_id)}${status}</option>`);
  });
  ["run-a", "run-b"].forEach(id => {
    const sel = document.getElementById(id);
    const prev = sel.value;
    sel.innerHTML = html.join("");
    if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
  });
  document.getElementById("filter-count").innerText =
    `showing ${filtered.length} of ${allRuns.length} runs`;
}

function shortId(rid) {
  // run_id ends with _<unix-timestamp>; show a human-readable time instead.
  const m = (rid || "").match(/_(\d{10})$/);
  if (!m) return rid || "?";
  return new Date(parseInt(m[1], 10) * 1000).toLocaleString();
}

// Populate the val-metric dropdown from the union of val series across A and B.
function populateMetricSelect() {
  const sel = document.getElementById("metric");
  const names = new Set();
  [runA, runB].forEach(run => {
    if (run && run.series && run.series.val)
      Object.keys(run.series.val).forEach(n => names.add(n));
  });
  const sorted = [...names].sort();
  const prev = sel.value;
  sel.innerHTML = sorted.map(n => `<option value="${n}">${n}</option>`).join("");
  if (sorted.includes(prev)) sel.value = prev;
  else if (sorted.includes("rel_l2")) sel.value = "rel_l2";
}

function fmt(x, digits = 4) {
  if (x == null || !Number.isFinite(x)) return "—";
  return Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(2) : x.toFixed(digits);
}

function valSeries(run, metric) {
  return run && run.series && run.series.val ? run.series.val[metric] : null;
}

function renderSummary(metric) {
  const target = document.getElementById("summary");
  if (!runA && !runB) { target.innerHTML = ""; return; }
  function cardOf(run, role) {
    if (!run) return `<div></div>`;
    const vs = valSeries(run, metric);
    const vv = vs ? vs.values.filter(Number.isFinite) : [];
    const ts = run.series && run.series.train ? run.series.train.loss : null;
    const tv = ts ? ts.values.filter(Number.isFinite) : [];
    const lastVal = vv.length ? vv[vv.length - 1] : null;
    const bestVal = vv.length ? Math.min(...vv) : null;
    const lastTrain = tv.length ? tv[tv.length - 1] : null;
    const pill = run.completed
      ? `<span class="pill ok">completed</span>`
      : `<span class="pill pending">in progress</span>`;
    const wt = run.wall_time_seconds;
    const wtStr = wt != null ? ` &middot; ${wt.toFixed(0)}s` : "";
    const nval = vs ? vs.steps.length : 0;
    return `<div>
      <span class="${role}-color">${role}</span>
      <code>${run.experiment} / ${run.optimizer} / seed=${run.seed}</code>
      ${pill}<br/>
      val points: <b>${nval}</b>${wtStr}<br/>
      ${metric}(last): <b>${fmt(lastVal)}</b> &nbsp;
      ${metric}(best): <b>${fmt(bestVal)}</b><br/>
      train_loss(last): <b>${fmt(lastTrain)}</b>
    </div>`;
  }
  target.innerHTML = cardOf(runA, "a") + cardOf(runB, "b");
}

function renderMeta() {
  const target = document.getElementById("meta");
  if (!runA && !runB) { target.innerHTML = ""; return; }
  const hpA = runA?.hyperparameters || {};
  const hpB = runB?.hyperparameters || {};
  const allKeys = new Set([...Object.keys(hpA), ...Object.keys(hpB)]);
  const rows = [...allKeys].sort().map(k => {
    const a = hpA[k] === undefined ? "—" : JSON.stringify(hpA[k]);
    const b = hpB[k] === undefined ? "—" : JSON.stringify(hpB[k]);
    const cls = (runA && runB && a !== b) ? "diff" : "";
    return `<tr class="${cls}"><td>${k}</td><td>${a}</td><td>${b}</td></tr>`;
  });
  target.innerHTML = `<table><thead><tr><th>key</th><th class="a-color">A</th><th class="b-color">B</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

async function refresh() {
  document.getElementById("status").innerText = "refreshing…";
  const runs = await listRuns();
  populateExpFilter(runs);
  populateSelects(runs);
  const a = document.getElementById("run-a").value;
  const b = document.getElementById("run-b").value;
  [runA, runB] = await Promise.all([a ? getRun(a) : null, b ? getRun(b) : null]);

  populateMetricSelect();
  const metric = document.getElementById("metric").value;
  const window = getSmoothingWindow();
  document.getElementById("train-window-label").innerText = window;
  document.getElementById("val-heading").innerText = "Val " + (metric || "metric");
  charts.val.options.scales.y.title.text = metric || "value";

  const valA = runA ? points(valSeries(runA, metric)) : [];
  const valB = runB ? points(valSeries(runB, metric)) : [];
  const trainA = runA ? decimate(rollingMean(runA.series?.train?.loss, window), 4000) : [];
  const trainB = runB ? decimate(rollingMean(runB.series?.train?.loss, window), 4000) : [];

  applyLogScale();
  setChart(charts.val, valA, valB);
  setChart(charts.train, trainA, trainB);

  renderSummary(metric);
  renderMeta();
  document.getElementById("status").innerText = "ok · " + new Date().toLocaleTimeString();
}

["run-a", "run-b", "since-filter", "exp-filter", "metric", "logy", "smoothing-window"]
  .forEach(id => document.getElementById(id).addEventListener("change", refresh));
document.getElementById("smoothing-window").addEventListener("input", () => {
  document.getElementById("train-window-label").innerText = getSmoothingWindow();
});

setInterval(() => {
  if (document.getElementById("auto-refresh").checked) refresh();
}, 2000);

refresh();
</script>
</body>
</html>
"""


# Listing metadata is tiny but our artifacts embed full per-step curves (up to
# hundreds of thousands of lines), so re-parsing every file on each 2s poll is
# the dominant cost. Extract listing metadata from just the first line (meta)
# plus a tail read (for the terminal `end` record), and cache by (path, mtime):
# a completed file never changes, and a live run changes only its own file.
_META_CACHE: dict[str, tuple[float, dict]] = {}
_META_CACHE_LOCK = threading.Lock()


def _first_meta(path: str) -> dict:
    """Parse just the first line — the ``meta`` record. O(1) in file size."""
    try:
        with open(path, "rb") as f:
            first = f.readline()
        rec = json.loads(first)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return rec if rec.get("type") == "meta" else {}


def _tail_end(path: str, n_bytes: int = 8192) -> dict | None:
    """Find the terminal ``end`` record by reading only the file's tail."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n_bytes))
            chunk = f.read()
    except OSError:
        return None
    for line in reversed(chunk.split(b"\n")):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if rec.get("type") == "end":
            return rec
    return None


def _extract_meta(full: str, rel: str) -> dict | None:
    """Listing metadata for one artifact; None if there's no meta line yet."""
    meta = _first_meta(full)
    if not meta:
        return None
    end = _tail_end(full)
    return {
        "path": rel,
        "experiment": meta.get("experiment", ""),
        "optimizer": meta.get("optimizer", ""),
        "seed": meta.get("seed", -1),
        "run_id": meta.get("run_id", ""),
        "completed": bool(end and end.get("completed")),
    }


def _load_run(path: str) -> dict:
    """Parse a full JSONL run into a chart-ready payload.

    ``series`` is ``{kind: {metric: {"steps": [...], "values": [...]}}}`` over
    the ``train`` / ``val`` event kinds. Non-finite metric values are dropped
    (they can't ride the log axis and aren't valid JSON anyway). Tolerant of a
    truncated final line so a live run loads fine.
    """
    meta: dict = {}
    end: dict | None = None
    series: dict[str, dict[str, dict]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            kind = rec.get("type")
            if kind == "meta":
                meta = rec
            elif kind == "end":
                end = rec
            elif kind:
                step = rec.get("step")
                per_kind = series.setdefault(kind, {})
                for name, val in (rec.get("metrics") or {}).items():
                    if not isinstance(val, (int, float)) or not math.isfinite(val):
                        continue
                    pair = per_kind.setdefault(name, {"steps": [], "values": []})
                    pair["steps"].append(step)
                    pair["values"].append(val)
    return {
        "experiment": meta.get("experiment", ""),
        "optimizer": meta.get("optimizer", ""),
        "seed": meta.get("seed", -1),
        "run_id": meta.get("run_id", ""),
        "hyperparameters": meta.get("hyperparameters", {}),
        "completed": bool(end and end.get("completed")),
        "wall_time_seconds": (end or {}).get("wall_time_seconds"),
        "series": series,
    }


class _Handler(BaseHTTPRequestHandler):
    runs_dir: str = "runs"

    def _send_json(self, status: int, payload) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client navigated away mid-poll — benign

    def _send_html(self, body: str) -> None:
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, fmt, *args) -> None:
        return  # silence the default per-request access log

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path == "/":
            self._send_html(INDEX_HTML)
            return
        if u.path == "/api/runs":
            self._send_json(200, self._list_runs())
            return
        if u.path == "/api/run":
            rel = parse_qs(u.query).get("path", [""])[0]
            try:
                self._send_json(200, _load_run(self._resolve(rel)))
            except Exception as e:
                self._send_json(404, {"error": str(e)})
            return
        self.send_response(404)
        self.end_headers()

    def _resolve(self, rel: str) -> str:
        """Reject anything outside the runs dir — no path traversal."""
        if not rel:
            raise FileNotFoundError("empty path")
        runs_abs = os.path.abspath(self.runs_dir)
        candidate = os.path.abspath(os.path.join(self.runs_dir, rel))
        if not (candidate == runs_abs or candidate.startswith(runs_abs + os.sep)):
            raise PermissionError(f"path outside runs dir: {rel}")
        return candidate

    def _list_runs(self) -> list[dict]:
        entries: list[dict] = []
        if not os.path.isdir(self.runs_dir):
            return entries
        seen: set[str] = set()
        for root, _dirs, files in os.walk(self.runs_dir):
            for fname in files:
                if not fname.endswith(".jsonl"):
                    continue
                full = os.path.join(root, fname)
                try:
                    mtime = os.path.getmtime(full)
                except OSError:
                    continue  # file vanished between walk and stat
                seen.add(full)
                with _META_CACHE_LOCK:
                    cached = _META_CACHE.get(full)
                if cached is not None and cached[0] == mtime:
                    meta = cached[1]
                else:
                    rel = os.path.relpath(full, self.runs_dir)
                    meta = _extract_meta(full, rel)
                    if meta is None:
                        continue  # half-written / no meta yet — skip silently
                    with _META_CACHE_LOCK:
                        _META_CACHE[full] = (mtime, meta)
                entries.append({**meta, "mtime": mtime})
        with _META_CACHE_LOCK:  # evict entries for deleted files
            for stale in _META_CACHE.keys() - seen:
                del _META_CACHE[stale]
        entries.sort(key=lambda r: r["mtime"], reverse=True)
        return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    _Handler.runs_dir = args.runs_dir
    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"viewer running at http://{args.host}:{args.port}  "
          f"(runs-dir={args.runs_dir})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
        httpd.server_close()


if __name__ == "__main__":
    main()
