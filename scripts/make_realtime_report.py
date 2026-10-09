"""Generate results/REALTIME.md from the latency_bench runs in results/realtime/runs (never hand-edit the tables).

Each run is <config>_r<k>.json (C++ summary) + <config>_r<k>.csv[.gz] (per-iteration rows). Configs are named
<interference>__<mitigation>. Repeats of a config are pooled for percentiles (numpy "linear", identical to the C++
rt_stats code); the per-repeat p99 range is shown so run-to-run noise is visible.

    python scripts/make_realtime_report.py [--runs results/realtime/runs] [--out results/REALTIME.md]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_RE = re.compile(r"^(?P<config>.+)_r(?P<rep>\d+)\.json$")

INTERFERENCE_DESC = {
    "idle": "no load besides the benchmark",
    "cpu": "32 spinning threads (`cpu_hog --mode cpu`), one per vCPU",
    "membw": "16 threads memcpy-ing 2x256 MiB buffers (`cpu_hog --mode membw`)",
    "gpu": "second process free-running the same TensorRT FP16 engine back to back (GPU co-tenant)",
    "gpuinproc": "synthetic FMA kernel (4640 x 256 threads, ~0.74 ms per launch) looping on a second CUDA stream inside the benchmark process",
    "gpuproc": "the same synthetic FMA kernel looping in a separate process (`gpu_hog`)",
    "combined": "16 cpu threads + 16 membw threads + the TensorRT co-tenant process",
}
MITIGATION_DESC = {
    "none": "none",
    "pin": "`taskset -c 15,31` for the benchmark (one physical core, both SMT siblings), all hogs/co-tenant on CPUs 0-14,16-30",
    "fifo": "`chrt -f 80` (SCHED_FIFO) for the benchmark process, no pinning",
    "pin_fifo": "pin + `chrt -f 80`",
    "pin_fifo_mlock": "pin + `chrt -f 80` + `mlockall(MCL_CURRENT|MCL_FUTURE)`",
    "cotenant_10hz": "co-tenant rate-limited to 10 Hz instead of free-running",
    "mps25": "CUDA MPS with the co-tenant limited to `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25`",
    "stream_prio": "inference stream created with the greatest CUDA stream priority (`cudaStreamCreateWithPriority`), hog stream at the least",
    "all": "pin + `chrt -f 80` + mlockall + MPS co-tenant limit 25 %",
    "pin_iso": "pin + cpuset isolation: system.slice and user.slice (all daemons, hogs, co-tenant) confined to CPUs 0-14,16-30 via cgroup v2 `AllowedCPUs`, benchmark in its own scope on 15,31",
    "all_iso": "all + the cpuset isolation of pin_iso",
}
# mitigation config -> unmitigated reference with the same interference
ORDER = [
    "idle__none", "cpu__none", "membw__none", "gpu__none", "gpuinproc__none", "gpuproc__none", "combined__none",
    "idle__pin_fifo_mlock", "cpu__pin", "cpu__fifo", "cpu__pin_fifo", "membw__pin", "membw__pin_fifo",
    "gpu__cotenant_10hz", "gpu__mps25", "gpuinproc__stream_prio", "gpuproc__stream_prio", "cpu__pin_iso", "combined__pin_fifo_mlock", "combined__all", "combined__all_iso",
]


def read_rows(json_path: Path) -> list[dict]:
    csv_path = json_path.with_suffix(".csv")
    gz_path = json_path.with_suffix(".csv.gz")
    if gz_path.exists():
        f = gzip.open(gz_path, "rt", newline="")
    elif csv_path.exists():
        f = open(csv_path, newline="")
    else:
        raise FileNotFoundError(f"no per-iteration CSV for {json_path}")
    with f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q * 100.0, method="linear"))


def load_runs(runs_dir: Path) -> dict[str, list[dict]]:
    """config -> list of {"summary": json, "rows": [...]} sorted by repeat."""
    out: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(runs_dir.glob("*.json")):
        m = RUN_RE.match(p.name)
        if not m:
            continue
        out[m["config"]].append({"rep": int(m["rep"]), "summary": json.loads(p.read_text()), "rows": read_rows(p)})
    for v in out.values():
        v.sort(key=lambda r: r["rep"])
    return dict(out)


def pool(runs: list[dict]) -> dict:
    """Pooled statistics over all repeats of one config."""
    col = lambda k: np.array([r[k] for run in runs for r in run["rows"]], dtype=np.float64)
    e2e, resp, late = col("e2e_ms"), col("response_ms"), col("wake_late_us")
    deadline = runs[0]["summary"]["config"]["deadline_ms"]
    res = {
        "n": int(e2e.size),
        "repeats": len(runs),
        "e2e": {q: pct(e2e, v) for q, v in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("p999", 0.999))},
        "e2e_max": float(e2e.max()),
        "e2e_sd": float(e2e.std(ddof=1)) if e2e.size > 1 else 0.0,
        "resp_p99": pct(resp, 0.99),
        "resp_max": float(resp.max()),
        "jitter_sd_us": float(late.std(ddof=1)) if late.size > 1 else 0.0,
        "jitter_p99_us": pct(late, 0.99),
        "jitter_max_us": float(late.max()),
        "misses": int((resp > deadline).sum()),
        "skipped": int(sum(run["summary"]["skipped_releases"] for run in runs)),
        "rep_p99": [run["summary"]["e2e_ms"]["p99"] for run in runs],
        "stages": {s: (pct(col(f"{s}_ms"), 0.5), pct(col(f"{s}_ms"), 0.99)) for s in ("pre", "h2d", "infer", "d2h")},
    }
    rs = [run["summary"]["resources"] for run in runs]
    mean = lambda k: float(np.mean([r[k] for r in rs if r.get(k) is not None])) if any(r.get(k) is not None for r in rs) else float("nan")
    res["res"] = {k: mean(k) for k in ("cpu_pct_of_one_core", "loop_thread_vol_csw", "loop_thread_invol_csw",
                                       "proc_invol_csw", "maxrss_mb", "gpu_util_mean_pct", "gpu_proc_mem_max_mb",
                                       "gpu_dev_mem_used_max_mb")}
    res["fifo"] = sorted({run["summary"]["config"]["fifo"] for run in runs})
    res["mlock"] = sorted({run["summary"]["config"]["mlock"] for run in runs})
    res["affinity"] = sorted({run["summary"]["config"]["affinity"] for run in runs})
    return res


LOG_PATTERNS = {
    "cotenant": re.compile(r"^n=(?P<n>\d+) rate=[\d.]+Hz e2e p50=(?P<p50>[\d.]+) p99=(?P<p99>[\d.]+)", re.M),
    "gpu_hog": re.compile(r"busy_fraction=(?P<busy>[\d.]+)"),
    "membw_hog": re.compile(r"GB_per_s=(?P<gbs>[\d.]+)"),
    "cpu_hog": re.compile(r"Mops_per_s=(?P<mops>[\d.]+)"),
}


def load_generator_logs(runs_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """config -> generator kind -> list of parsed numbers (one per repeat) from <config>_r<k>.<kind>.log."""
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for p in sorted(runs_dir.glob("*.log")):
        m = re.match(r"^(?P<config>.+)_r\d+\.(?P<kind>cotenant|gpu_hog|membw_hog|cpu_hog)\.log$", p.name)
        if not m:
            continue
        hit = LOG_PATTERNS[m["kind"]].search(p.read_text(errors="replace"))
        if hit:
            out[m["config"]][m["kind"]].append({k: float(v) for k, v in hit.groupdict().items()})
    return out


def verdict(ref: dict, mit: dict) -> str:
    """'helped' / 'hurt' only when the pooled p99 moves >= 5 % AND the per-repeat p99 ranges do not overlap."""
    r, m = ref["e2e"]["p99"], mit["e2e"]["p99"]
    if m <= 0.95 * r and max(mit["rep_p99"]) < min(ref["rep_p99"]):
        return "helped"
    if m >= 1.05 * r and min(mit["rep_p99"]) > max(ref["rep_p99"]):
        return "hurt"
    return "no clear effect"


def fmt(x: float, nd: int = 2) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def build_report(runs: dict[str, list[dict]], host: dict | None = None, gen_logs: dict | None = None) -> str:
    stats = {c: pool(r) for c, r in runs.items()}
    configs = [c for c in ORDER if c in stats] + sorted(c for c in stats if c not in ORDER)
    any_run = next(iter(runs.values()))[0]["summary"]
    cfg, h = any_run["config"], any_run["host"]
    L: list[str] = []
    L.append("# Real-time latency and jitter under interference")
    L.append("")
    L.append("_Generated by `scripts/make_realtime_report.py` from `results/realtime/runs/` (do not edit by hand)._")
    L.append("")
    L.append(f"**Platform: AWS EC2 {h['gpu']} VM ({h['nproc']} vCPU AMD EPYC 7R13), not an industrial edge device, "
             f"not PREEMPT_RT, hypervisor noise present.** Kernel `{h['kernel']}` (`{h['kernel_version']}`)"
             + (f"; preemption model: {host['preempt']}" if host and host.get("preempt") else "") + ". "
             "No `isolcpus`/`nohz_full`, IRQs not steered. Numbers are measured on this VM only and do not transfer to a "
             "Jetson/IPC target without re-measuring.")
    L.append("")
    L.append("## What is measured")
    L.append("")
    L.append(f"`cpp/bench/latency_bench` runs the BEVFusion TensorRT FP16 engine (`results/export/bevfusion_fp16.engine`, "
             f"TensorRT 10.16) in a periodic loop at **{cfg['rate_hz']:g} Hz** (absolute `clock_nanosleep(CLOCK_MONOTONIC)` "
             f"releases, deadline = period = {cfg['deadline_ms']:.2f} ms). One job = host staging copy of the "
             f"{cfg['input_bytes'] / 1e6:.1f} MB input frame (pageable -> pinned, stand-in for pre-processing; voxelisation "
             f"and LSS indexing are not ported to C++), H2D copies, `enqueueV3`, D2H of all outputs "
             f"({cfg['output_bytes'] / 1e6:.1f} MB incl. the occupancy volume), stream synchronise.")
    L.append("")
    L.append("* **e2e** = wake-up to outputs on the host; **response** = planned release to outputs on the host.")
    L.append("* **jitter** = wake lateness (actual wake - planned release); sd and max over all pooled iterations.")
    L.append("* **miss** = response > deadline. Overruns follow a frame-drop policy (stale releases skipped, counted).")
    L.append(f"* Each config: {stats[configs[0]]['repeats']} repeats x {cfg['iters']} measured iterations "
             f"({cfg['warmup']} warm-up iterations excluded); percentiles pooled over repeats (numpy linear = C++ rt_stats).")
    L.append("* p99 range = min..max of the per-repeat p99, to show run-to-run noise.")
    L.append("")
    L.append("Interference generators: " + "; ".join(f"**{k}**: {v}" for k, v in INTERFERENCE_DESC.items()) + ".")
    L.append("")
    L.append("## 1. All configs (e2e latency, ms)")
    L.append("")
    L.append("| config | interference | mitigation | n | p50 | p99 | p99.9 | max | p99 range | jitter sd us | jitter max us | misses | skipped |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in configs:
        s = stats[c]
        kind, mit = (c.split("__", 1) + ["?"])[:2]
        L.append(f"| `{c}` | {kind} | {mit} | {s['n']} | {fmt(s['e2e']['p50'])} | {fmt(s['e2e']['p99'])} | "
                 f"{fmt(s['e2e']['p999'])} | {fmt(s['e2e_max'])} | {fmt(min(s['rep_p99']))}..{fmt(max(s['rep_p99']))} | "
                 f"{fmt(s['jitter_sd_us'], 1)} | {fmt(s['jitter_max_us'], 1)} | {s['misses']} | {s['skipped']} |")
    L.append("")
    L.append("## 2. Mitigations vs the same interference without mitigation")
    L.append("")
    L.append("Verdict rule (automatic): **helped** / **hurt** only if pooled p99 moves by >= 5 % and the per-repeat p99 "
             "ranges do not overlap; otherwise **no clear effect**.")
    L.append("")
    L.append("| mitigation config | reference | p99 ref -> mit | p99.9 ref -> mit | max ref -> mit | jitter sd us ref -> mit | misses ref -> mit | verdict |")
    L.append("|---|---|---|---|---|---|---|---|")
    findings = []
    for c in configs:
        kind, mit = (c.split("__", 1) + ["?"])[:2]
        ref = f"{kind}__none"
        if mit == "none" or ref not in stats:
            continue
        r, m = stats[ref], stats[c]
        v = verdict(r, m)
        d99 = 100.0 * (m["e2e"]["p99"] - r["e2e"]["p99"]) / r["e2e"]["p99"]
        findings.append((c, ref, v, d99))
        L.append(f"| `{c}` | `{ref}` | {fmt(r['e2e']['p99'])} -> {fmt(m['e2e']['p99'])} ({d99:+.1f} %) | "
                 f"{fmt(r['e2e']['p999'])} -> {fmt(m['e2e']['p999'])} | {fmt(r['e2e_max'])} -> {fmt(m['e2e_max'])} | "
                 f"{fmt(r['jitter_sd_us'], 1)} -> {fmt(m['jitter_sd_us'], 1)} | {r['misses']} -> {m['misses']} | **{v}** |")
    L.append("")
    L.append("Mitigation definitions: " + "; ".join(f"**{k}**: {v}" for k, v in MITIGATION_DESC.items() if k != "none") + ".")
    L.append("")
    L.append("## 3. Where the time goes (p50 / p99 per stage, ms)")
    L.append("")
    L.append("`pre` = host staging memcpy (CPU wall time); `h2d`, `infer`, `d2h` = cudaEvent GPU time on the inference stream.")
    L.append("")
    L.append("| config | pre | h2d | infer | d2h |")
    L.append("|---|---|---|---|---|")
    for c in configs:
        st = stats[c]["stages"]
        L.append(f"| `{c}` | " + " | ".join(f"{fmt(st[k][0])} / {fmt(st[k][1])}" for k in ("pre", "h2d", "infer", "d2h")) + " |")
    L.append("")
    L.append("## 4. Resource usage (mean over repeats, measured window only)")
    L.append("")
    L.append("CPU % is process CPU time / wall time (100 % = one core; the CUDA sync spin-waits by default). csw = context "
             "switches of the loop thread (getrusage RUSAGE_THREAD). GPU util / memory from NVML sampled every 50 ms; GPU "
             "util is device-wide and includes any co-tenant.")
    L.append("")
    L.append("| config | CPU % of 1 core | loop vol csw | loop invol csw | max RSS MB | GPU util % | GPU mem (proc) MB | GPU mem (device) MB | SCHED_FIFO | mlock |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in configs:
        r = stats[c]["res"]
        L.append(f"| `{c}` | {fmt(r['cpu_pct_of_one_core'], 1)} | {fmt(r['loop_thread_vol_csw'], 0)} | "
                 f"{fmt(r['loop_thread_invol_csw'], 0)} | {fmt(r['maxrss_mb'], 0)} | {fmt(r['gpu_util_mean_pct'], 1)} | "
                 f"{fmt(r['gpu_proc_mem_max_mb'], 0)} | {fmt(r['gpu_dev_mem_used_max_mb'], 0)} | "
                 f"{', '.join(stats[c]['fifo'])} | {', '.join(stats[c]['mlock'])} |")
    L.append("")
    if gen_logs:
        L.append("## 5. What the interference generators achieved (mean over repeats)")
        L.append("")
        L.append("Co-tenant = the second TensorRT process (its own per-inference latency, i.e. the price the co-tenant pays "
                 "for a mitigation); membw GB/s = bytes moved by `cpu_hog --mode membw`; GPU hog busy = fraction of wall "
                 "time with a hog kernel in flight.")
        L.append("")
        L.append("| config | co-tenant inferences | co-tenant p50 ms | co-tenant p99 ms | cpu_hog Mops/s | membw GB/s | GPU hog busy |")
        L.append("|---|---|---|---|---|---|---|")
        avg = lambda xs, k: float(np.mean([x[k] for x in xs])) if xs else float("nan")
        for c in configs:
            g = gen_logs.get(c)
            if not g:
                continue
            co, gh, mb, ch = g.get("cotenant", []), g.get("gpu_hog", []), g.get("membw_hog", []), g.get("cpu_hog", [])
            L.append(f"| `{c}` | {fmt(avg(co, 'n'), 0)} | {fmt(avg(co, 'p50'))} | {fmt(avg(co, 'p99'))} | "
                     f"{fmt(avg(ch, 'mops'), 0)} | {fmt(avg(mb, 'gbs'), 1)} | {fmt(avg(gh, 'busy'), 3)} |")
        L.append("")
    L.append("## 6. Findings (from the tables above)")
    L.append("")
    base = stats.get("idle__none")
    for c in ("cpu__none", "membw__none", "gpu__none", "gpuinproc__none", "gpuproc__none", "combined__none"):
        if base and c in stats:
            s = stats[c]
            L.append(f"* `{c}`: p99 {fmt(base['e2e']['p99'])} -> {fmt(s['e2e']['p99'])} ms "
                     f"({100.0 * (s['e2e']['p99'] - base['e2e']['p99']) / base['e2e']['p99']:+.1f} % vs idle), "
                     f"max {fmt(s['e2e_max'])} ms, {s['misses']} misses / {s['n']}.")
    for c, ref, v, d99 in findings:
        extra = f"; p50 {fmt(stats[c]['e2e']['p50'])} ms vs idle {fmt(base['e2e']['p50'])} ms" if base else ""
        L.append(f"* `{c}` vs `{ref}`: **{v}** (p99 {d99:+.1f} %, misses {stats[ref]['misses']} -> {stats[c]['misses']}{extra}).")
    L.append("")
    if "cpu__pin" in stats and "cpu__none" in stats and verdict(stats["cpu__none"], stats["cpu__pin"]) == "hurt":
        L.append(f"Why `cpu__pin` hurt: confining only the hogs (taskset) to the other 30 CPUs leaves CPUs 15/31 as the only "
                 f"idle CPUs, so the scheduler migrates every other runnable task there. A separate diagnostic run "
                 f"(`ps -eLo psr` sampling, not part of the tables) showed s3fs mount daemons and other system.slice threads "
                 f"running on 15/31 next to the benchmark; the loop thread's involuntary context switches rose from "
                 f"{fmt(stats['cpu__none']['res']['loop_thread_invol_csw'], 0)} to "
                 f"{fmt(stats['cpu__pin']['res']['loop_thread_invol_csw'], 0)} per run. Pinning only works together with "
                 f"SCHED_FIFO (`cpu__pin_fifo`) or real isolation of everything else (`cpu__pin_iso`, cgroup cpuset).")
        L.append("")
    if base and "membw__pin_fifo" in stats:
        m = stats["membw__pin_fifo"]
        L.append(f"Memory bandwidth is not fixed by any CPU-side mitigation: under `membw__pin_fifo` the deadline tail is gone, "
                 f"but the host staging copy still takes {fmt(m['stages']['pre'][0])} ms p50 vs {fmt(base['stages']['pre'][0])} ms "
                 f"idle and e2e p50 stays at {fmt(m['e2e']['p50'])} ms vs {fmt(base['e2e']['p50'])} ms (shared DRAM / L3; would need "
                 f"memory-bandwidth throttling of the noisy neighbour (AMD PQoS / Intel MBA; this VM shows no rdt_a/mba CPU flags and no resctrl filesystem) or "
                 f"less host copying).")
        L.append("")
    if "combined__all" in stats and "combined__all_iso" in stats:
        a, b = stats["combined__all"], stats["combined__all_iso"]
        L.append(f"Under combined load the best configs still miss: `combined__all` {a['misses']} / {a['n']}, "
                 f"`combined__all_iso` {b['misses']} / {b['n']} (p99 {fmt(a['e2e']['p99'])} vs {fmt(b['e2e']['p99'])} ms); the "
                 f"remaining tail comes from the memory-bandwidth slowdown of the staging copy plus the GPU share the "
                 f"MPS-limited co-tenant still takes, which cpuset isolation does not address.")
        L.append("")
    L.append("## Limitations")
    L.append("")
    L.append("* Cloud VM (AWS EC2, NVIDIA L4, AMD EPYC 7R13): vCPUs are hypervisor threads, steal time and host interrupts "
             "are outside our control; not an embedded/edge device and not a PREEMPT_RT kernel. Absolute numbers and "
             "tails will differ on Jetson Orin or an industrial PC.")
    L.append("* No `isolcpus`, `nohz_full`, `rcu_nocbs` or IRQ affinity (would need a reboot with new kernel parameters); "
             "pinning uses taskset only, so kernel threads and IRQs can still run on the inference core.")
    L.append("* One input frame replayed every iteration (static shapes, so per-frame content does not change the work); "
             "pre-processing is a staging memcpy, not the real voxelisation/LSS indexing.")
    L.append("* CUDA stream priorities only act inside one CUDA context; they cannot prioritise against another process "
             "(that case is covered by MPS and co-tenant rate limiting instead). GPU time-slicing between processes "
             "without MPS is controlled by the driver and cannot be tuned here.")
    L.append("* MPS was started only for the `mps25` / `all` / `all_iso` configs; under MPS the benchmark itself is also an MPS "
             "client. With 25 % of the SMs (14 of 58) TensorRT warns that the co-tenant engine was built for 58 SMs and that "
             "deadlocks are possible; no deadlock occurred in these runs, but a production setup would build the co-tenant "
             "engine for its SM budget.")
    L.append("* The cpuset isolation is done at runtime with `systemctl set-property --runtime ... AllowedCPUs` (cgroup v2) "
             "and undone after the config; kernel threads and IRQs are not moved by it.")
    L.append(f"* {stats[configs[0]]['repeats']} repeats per config: enough to see repeat-to-repeat spread, not enough for tight confidence intervals on p99.9 "
             "and max (pooled n per config is shown).")
    L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "results/realtime/runs"))
    ap.add_argument("--out", default=str(ROOT / "results/REALTIME.md"))
    a = ap.parse_args()
    runs_dir = Path(a.runs)
    runs = load_runs(runs_dir)
    if not runs:
        raise SystemExit(f"no runs in {runs_dir}")
    host_path = runs_dir.parent / "host.json"
    host = json.loads(host_path.read_text()) if host_path.exists() else None
    Path(a.out).write_text(build_report(runs, host, load_generator_logs(runs_dir)))
    print(f"wrote {a.out} from {sum(len(v) for v in runs.values())} runs / {len(runs)} configs")


if __name__ == "__main__":
    main()
