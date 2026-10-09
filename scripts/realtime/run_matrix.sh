#!/usr/bin/env bash
# Real-time latency/jitter matrix: interference scenarios x mitigations for cpp/bench/latency_bench.
# Every run writes <out>/<config>_r<k>.{csv,json}; scripts/make_realtime_report.py turns them into results/REALTIME.md.
#
#   BUILD=<cmake build dir> TRT_LIBS=<dir with libnvinfer.so.10> scripts/realtime/run_matrix.sh [config ...]
#
# Needs sudo (SCHED_FIFO, mlockall, MPS daemon). Interference generators: build/cpu_hog (cpu, membw) and a second
# latency_bench process free-running the same TensorRT engine (GPU co-tenant). Placement for the pinned configs:
# inference on RT_CPUS (one physical core = two SMT siblings), every hog/co-tenant thread on OTHER_CPUS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD=${BUILD:?set BUILD to the cmake build dir}
TRT_LIBS=${TRT_LIBS:?set TRT_LIBS to the TensorRT lib dir}
cd "$ROOT"  # relative engine/frame paths end up in the JSON summaries
ENGINE=${ENGINE:-results/export/bevfusion_fp16.engine}
FRAME=${FRAME:-assets/samples/00_c923fe08}
OUT=${OUT:-$ROOT/results/realtime/runs}
ITERS=${ITERS:-2000}
WARMUP=${WARMUP:-100}
RATE=${RATE:-30}
REPEATS=${REPEATS:-3}
RT_CPUS=${RT_CPUS:-15,31}
OTHER_CPUS=${OTHER_CPUS:-0-14,16-30}
FIFO_PRIO=${FIFO_PRIO:-80}
MPS_PCT=${MPS_PCT:-25}
HOG_BLOCKS=${HOG_BLOCKS:-4640}  # synthetic GPU hog: 4640 x 256 threads, ~0.74 ms per kernel on the L4
HOG_INNER=${HOG_INNER:-2000}    # FMA iterations per thread (short blocks so the block scheduler can interleave)
mkdir -p "$OUT"
BENCH="$BUILD/latency_bench"
PIDS=()

ALL_CONFIGS=(
  idle__none cpu__none membw__none gpu__none combined__none
  idle__pin_fifo_mlock
  cpu__pin cpu__fifo cpu__pin_fifo
  membw__pin membw__pin_fifo
  gpu__cotenant_10hz gpu__mps25
  gpuinproc__none gpuinproc__stream_prio gpuproc__none gpuproc__stream_prio
  combined__pin_fifo_mlock combined__all
  cpu__pin_iso combined__all_iso
)

start_bg() {  # start_bg <pinned 0|1> <sudo 0|1> <log> cmd...
  local pinned=$1 su=$2 log=$3; shift 3
  local pre=()
  [[ $su == 1 ]] && pre=(sudo env "LD_LIBRARY_PATH=$TRT_LIBS" "${MPS_ENV[@]}")
  [[ $pinned == 1 ]] && pre+=(taskset -c "$OTHER_CPUS")
  "${pre[@]}" "$@" >"$log" 2>&1 &
  PIDS+=($!)
}

start_interference() {  # <kind> <pinned> <log prefix> <cotenant rate>
  local kind=$1 pinned=$2 log=$3 corate=$4
  local cotenant=("$BENCH" --engine "$ENGINE" --frame "$FRAME" --rate "$corate" --iters 100000000 --warmup 0)
  case $kind in
    idle) ;;
    cpu) start_bg "$pinned" 0 "$log.cpu_hog.log" "$BUILD/cpu_hog" --mode cpu --threads 32 ;;
    membw) start_bg "$pinned" 0 "$log.membw_hog.log" "$BUILD/cpu_hog" --mode membw --threads 16 ;;
    gpu) start_bg "$pinned" 1 "$log.cotenant.log" env "${CO_ENV[@]}" "${cotenant[@]}" ;;
    gpuinproc) ;;  # handled inside the measured process (--inproc-hog)
    gpuproc) start_bg "$pinned" 0 "$log.gpu_hog.log" "$BUILD/gpu_hog" --inner "$HOG_INNER" --blocks "$HOG_BLOCKS" --priority least ;;
    combined)
      start_bg "$pinned" 0 "$log.cpu_hog.log" "$BUILD/cpu_hog" --mode cpu --threads 16
      start_bg "$pinned" 0 "$log.membw_hog.log" "$BUILD/cpu_hog" --mode membw --threads 16
      start_bg "$pinned" 1 "$log.cotenant.log" env "${CO_ENV[@]}" "${cotenant[@]}" ;;
    *) echo "unknown interference $kind"; exit 2 ;;
  esac
  [[ $kind == idle || $kind == gpuinproc ]] || sleep 5  # let hogs / co-tenant engine load reach steady state
}

stop_interference() {
  # children run under sudo for the co-tenant: signal the whole process tree by name
  # bracketed patterns so pkill (and its sudo parent) never match their own command line
  pkill -TERM -f "cpu_ho[g] --mode" 2>/dev/null || true
  pkill -TERM -f "gpu_ho[g] --inner" 2>/dev/null || true
  sudo pkill -TERM -f "[-]-iters 100000000" 2>/dev/null || true
  for p in "${PIDS[@]:-}"; do [[ -n $p ]] && wait "$p" 2>/dev/null || true; done
  PIDS=()
}

mps_start() {
  sudo nvidia-cuda-mps-control -d
  sleep 1
}
mps_stop() { echo quit | sudo nvidia-cuda-mps-control || true; sleep 2; }

# cpuset isolation (cgroup v2 via systemd, runtime only): every other task (system.slice, user.slice incl. the hogs)
# is confined to OTHER_CPUS; the benchmark runs in its own scope on RT_CPUS. Undone after the config.
iso_on() {
  sudo systemctl set-property --runtime system.slice "AllowedCPUs=$OTHER_CPUS"
  sudo systemctl set-property --runtime user.slice "AllowedCPUs=$OTHER_CPUS"
}
iso_off() {
  sudo systemctl set-property --runtime system.slice "AllowedCPUs=0-$(($(getconf _NPROCESSORS_CONF) - 1))" || true
  sudo systemctl set-property --runtime user.slice "AllowedCPUs=0-$(($(getconf _NPROCESSORS_CONF) - 1))" || true
  sudo rm -f /run/systemd/system.control/system.slice.d/50-AllowedCPUs.conf \
    /run/systemd/system.control/user.slice.d/50-AllowedCPUs.conf
  sudo systemctl daemon-reload || true
}

run_config() {
  local cfg=$1 kind=${1%%__*} mit=${1#*__}
  local corate=0 pinned=0 iso=0 pre=() extra=()
  MPS_ENV=() CO_ENV=()
  case $mit in
    none) ;;
    pin) pinned=1 ;;
    fifo) pre=(chrt -f "$FIFO_PRIO") ;;
    pin_fifo) pinned=1; pre=(chrt -f "$FIFO_PRIO") ;;
    pin_fifo_mlock) pinned=1; pre=(chrt -f "$FIFO_PRIO"); extra=(--mlock) ;;
    cotenant_10hz) corate=10 ;;
    mps25) CO_ENV=("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PCT") ;;
    stream_prio) extra=(--stream-prio high) ;;
    all) pinned=1; pre=(chrt -f "$FIFO_PRIO"); extra=(--mlock); CO_ENV=("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PCT") ;;
    pin_iso) pinned=1; iso=1 ;;
    all_iso) pinned=1; iso=1; pre=(chrt -f "$FIFO_PRIO"); extra=(--mlock); CO_ENV=("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PCT") ;;
    *) echo "unknown mitigation $mit"; exit 2 ;;
  esac
  [[ ${#CO_ENV[@]} -eq 0 ]] && CO_ENV=("OBF_NOP=1")
  [[ $kind == gpuinproc ]] && extra+=(--inproc-hog "$HOG_BLOCKS" --hog-inner "$HOG_INNER" --inproc-hog-prio least)
  local use_mps=0
  [[ $mit == mps25 || $mit == all || $mit == all_iso ]] && use_mps=1
  [[ $use_mps == 1 ]] && mps_start
  [[ $iso == 1 ]] && iso_on
  for r in $(seq 1 "$REPEATS"); do
    local base="$OUT/${cfg}_r$r"
    start_interference "$kind" "$pinned" "$base" "$corate"
    local bench_pre=(sudo env "LD_LIBRARY_PATH=$TRT_LIBS")
    [[ $iso == 1 ]] && bench_pre=(sudo systemd-run --quiet --scope --slice=rtbench.slice -p "AllowedCPUs=$RT_CPUS"
      env "LD_LIBRARY_PATH=$TRT_LIBS")
    [[ $pinned == 1 ]] && bench_pre+=(taskset -c "$RT_CPUS")
    echo "== $cfg r$r"
    "${bench_pre[@]}" "${pre[@]}" "$BENCH" --engine "$ENGINE" --frame "$FRAME" --rate "$RATE" --iters "$ITERS" \
      --warmup "$WARMUP" --out "$base" "${extra[@]}" --meta "config=$cfg" --meta "interference=$kind" \
      --meta "mitigation=$mit" --meta "repeat=$r" --meta "mps=$use_mps" --meta "pinned=$pinned" --meta "cpuset_iso=$iso" \
      2> >(grep -v '^\[TRT\]' >&2)
    stop_interference
    sleep 2
  done
  [[ $use_mps == 1 ]] && mps_stop
  [[ $iso == 1 ]] && iso_off
  sudo chown -R "$(id -u):$(id -g)" "$OUT"
}

write_host() {  # host facts for the report header
  local preempt rt
  preempt=$(sudo cat /sys/kernel/debug/sched/preempt 2>/dev/null || echo unknown)
  rt=$([[ -e /sys/kernel/realtime ]] && cat /sys/kernel/realtime || echo absent)
  python3 - "$OUT/../host.json" "$preempt" "$rt" "$TRT_LIBS" <<'PY'
import json, platform, subprocess, sys, glob, os
out, preempt, rt, trt = sys.argv[1:5]
cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
libs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(trt, "libnvinfer.so.*")))
json.dump({"kernel": platform.release(), "preempt": "PREEMPT_DYNAMIC kernel, active mode " + preempt.strip(),
           "preempt_rt": "absent" if rt == "absent" else rt, "cpu": cpu, "gpu_driver": smi,
           "tensorrt_lib": libs}, open(out, "w"), indent=1)
PY
}

trap 'stop_interference; mps_stop >/dev/null 2>&1 || true; iso_off >/dev/null 2>&1 || true' EXIT
CONFIGS=("$@")
[[ ${#CONFIGS[@]} -eq 0 ]] && CONFIGS=("${ALL_CONFIGS[@]}")
write_host
for c in "${CONFIGS[@]}"; do run_config "$c"; done
