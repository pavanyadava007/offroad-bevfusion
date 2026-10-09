"""Unit tests for scripts/make_realtime_report.py (pooling, percentiles, verdict rule) on synthetic run files."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "make_realtime_report", Path(__file__).resolve().parents[1] / "scripts" / "make_realtime_report.py")
rr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rr)

COLS = ["iter", "release_ns", "wake_late_us", "pre_ms", "h2d_ms", "infer_ms", "d2h_ms", "e2e_ms", "response_ms", "skipped_after"]


def _write_run(d: Path, config: str, rep: int, e2e: np.ndarray, late_us: np.ndarray, deadline: float = 33.3):
    rows = []
    for i, (e, w) in enumerate(zip(e2e, late_us)):
        rows.append([i, int(i * 33.3e6), w, 1.0, 1.0, e - 3.0, 1.0, e, e + w / 1e3, 0])
    with open(d / f"{config}_r{rep}.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(COLS)
        wr.writerows(rows)
    s = lambda v: {"n": len(v), "mean": float(np.mean(v)), "sd": 0.0, "min": float(np.min(v)), "p50": float(np.percentile(v, 50)),
                   "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)), "p999": float(np.percentile(v, 99.9)),
                   "max": float(np.max(v))}
    summary = {
        "meta": {"config": config}, "host": {"kernel": "6.1-test", "kernel_version": "#1 SMP PREEMPT_DYNAMIC", "gpu": "TestGPU", "nproc": 4},
        "config": {"rate_hz": 30, "deadline_ms": deadline, "iters": len(e2e), "warmup": 10, "input_bytes": 1e6, "output_bytes": 2e6,
                   "affinity": "inherited", "fifo": "off", "mlock": "off"},
        "e2e_ms": s(e2e), "response_ms": s(e2e), "wake_late_us": s(late_us), "deadline_misses": 0, "skipped_releases": 0,
        "resources": {"cpu_pct_of_one_core": 50.0, "loop_thread_vol_csw": 10, "loop_thread_invol_csw": 1, "proc_invol_csw": 2,
                      "maxrss_mb": 100.0, "gpu_util_mean_pct": 40.0, "gpu_proc_mem_max_mb": 600.0, "gpu_dev_mem_used_max_mb": 900.0},
    }
    (d / f"{config}_r{rep}.json").write_text(json.dumps(summary))


def test_pool_matches_numpy_and_counts_misses(tmp_path):
    rng = np.random.default_rng(0)
    a, b = 20 + rng.random(500), 20 + rng.random(500)
    b[7] = 40.0  # one deadline miss (response = e2e + wake lateness > 33.3)
    late = np.full(500, 50.0)
    _write_run(tmp_path, "idle__none", 1, a, late)
    _write_run(tmp_path, "idle__none", 2, b, late)
    runs = rr.load_runs(tmp_path)
    s = rr.pool(runs["idle__none"])
    both = np.concatenate([a, b])
    assert s["n"] == 1000 and s["repeats"] == 2
    assert abs(s["e2e"]["p99"] - np.percentile(both, 99)) < 1e-3  # CSV keeps 4 decimals
    assert abs(s["e2e_max"] - 40.0) < 1e-9
    assert s["misses"] == 1
    assert s["jitter_sd_us"] == 0.0


def test_verdict_rule():
    ref = {"e2e": {"p99": 40.0}, "rep_p99": [39.0, 40.0, 41.0]}
    assert rr.verdict(ref, {"e2e": {"p99": 30.0}, "rep_p99": [29.0, 30.0, 31.0]}) == "helped"
    assert rr.verdict(ref, {"e2e": {"p99": 50.0}, "rep_p99": [49.0, 50.0, 51.0]}) == "hurt"
    # 5 % better pooled but repeat ranges overlap -> not claimed
    assert rr.verdict(ref, {"e2e": {"p99": 37.0}, "rep_p99": [33.0, 37.0, 39.5]}) == "no clear effect"
    assert rr.verdict(ref, {"e2e": {"p99": 39.5}, "rep_p99": [39.4, 39.5, 39.6]}) == "no clear effect"


def test_build_report_contains_labels_and_mitigation_rows(tmp_path):
    late = np.full(200, 10.0)
    for r in (1, 2, 3):
        _write_run(tmp_path, "cpu__none", r, np.full(200, 30.0 + r), late)
        _write_run(tmp_path, "cpu__pin", r, np.full(200, 20.0 + r * 0.1), late)
    md = rr.build_report(rr.load_runs(tmp_path), {"preempt": "PREEMPT_DYNAMIC kernel, active mode none voluntary (full)"})
    assert "not PREEMPT_RT" in md and "hypervisor noise present" in md
    assert "| `cpu__pin` | `cpu__none` |" in md and "**helped**" in md
    assert "\u2014" not in md and "\u2013" not in md  # plain hyphens only (no em/en dashes)
