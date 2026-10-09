// latency_bench: periodic real-time inference benchmark for the BEVFusion TensorRT engine.
//
// A fixed-rate loop (absolute deadlines, clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)) runs one full job per
// release: host staging copy of the input frame (pageable -> pinned, stands in for pre-processing), H2D copies,
// TensorRT enqueueV3, D2H copies of every output, stream synchronise. Per iteration it records wake lateness
// (release jitter), end-to-end latency (wake -> outputs on host), response time (release -> outputs on host) and the
// GPU-side split (cudaEvents). Deadline miss = response time > deadline. Overruns use a frame-drop policy (stale
// releases are skipped and counted). Resource usage: getrusage (process + loop thread), /proc RSS, NVML sampling.
//
//   latency_bench --engine E --frame DIR [--rate 30] [--deadline-ms 1000/rate] [--iters 2000] [--warmup 100]
//                 [--out PREFIX] [--cpus 15,31] [--fifo 80] [--mlock] [--stream-prio default|high|low]
//                 [--timerslack-ns N] [--blocking-sync] [--inproc-hog BLOCKS] [--inproc-hog-prio least|greatest] [--hog-inner N]
//                 [--meta key=value]...
// --rate 0 = free-running (no sleep), used for the TensorRT co-tenant process.
#include <cuda_runtime.h>
#include <nvml.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "hog_kernel.cuh"
#include "rt_stats.hpp"
#include "trt_runner.hpp"

#define CK(x)                                                                                                   \
  do {                                                                                                          \
    cudaError_t e_ = (x);                                                                                       \
    if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_));           \
  } while (0)

namespace {

std::atomic<bool> g_stop{false};
void on_sig(int) { g_stop = true; }

int64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

void sleep_until(int64_t t_ns) {
  timespec ts{static_cast<time_t>(t_ns / 1000000000LL), static_cast<long>(t_ns % 1000000000LL)};
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
  }
}

// Reads <dir>/<name>.npy (little-endian, C order; header skipped) or <dir>/<name>.bin (raw) and checks the size.
std::vector<char> read_tensor(const std::string& dir, const std::string& name, size_t expect) {
  std::ifstream f(dir + "/" + name + ".npy", std::ios::binary);
  bool npy = static_cast<bool>(f);
  if (!npy) f.open(dir + "/" + name + ".bin", std::ios::binary);
  if (!f) throw std::runtime_error("missing input " + dir + "/" + name + ".npy|.bin");
  std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  size_t off = 0;
  if (npy) {
    if (b.size() < 10 || std::memcmp(b.data(), "\x93NUMPY", 6) != 0) throw std::runtime_error(name + ": bad npy");
    const int major = static_cast<unsigned char>(b[6]);
    if (major == 1) off = 10 + (static_cast<unsigned char>(b[8]) | (static_cast<unsigned char>(b[9]) << 8));
    else off = 12 + (static_cast<uint32_t>(static_cast<unsigned char>(b[8])) |
                     (static_cast<uint32_t>(static_cast<unsigned char>(b[9])) << 8) |
                     (static_cast<uint32_t>(static_cast<unsigned char>(b[10])) << 16) |
                     (static_cast<uint32_t>(static_cast<unsigned char>(b[11])) << 24));
    const std::string hdr(b.data() + 10, off - 10);
    if (hdr.find("'fortran_order': True") != std::string::npos) throw std::runtime_error(name + ": fortran order");
  }
  if (b.size() - off != expect)
    throw std::runtime_error(name + ": " + std::to_string(b.size() - off) + " bytes != engine " + std::to_string(expect));
  return std::vector<char>(b.begin() + static_cast<long>(off), b.end());
}

std::vector<int> parse_cpus(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    auto dash = tok.find('-');
    if (dash == std::string::npos) out.push_back(std::stoi(tok));
    else for (int c = std::stoi(tok.substr(0, dash)); c <= std::stoi(tok.substr(dash + 1)); ++c) out.push_back(c);
  }
  return out;
}

double rss_mb() {
  std::ifstream f("/proc/self/statm");
  long pages_total = 0, pages_res = 0;
  f >> pages_total >> pages_res;
  return static_cast<double>(pages_res) * static_cast<double>(sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
}

double tv_s(const timeval& t) { return static_cast<double>(t.tv_sec) + static_cast<double>(t.tv_usec) / 1e6; }

std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o;
}

std::string summary_json(const obf::rt::Summary& s) {
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "{\"n\": %zu, \"mean\": %.4f, \"sd\": %.4f, \"min\": %.4f, \"p50\": %.4f, \"p95\": %.4f, "
                "\"p99\": %.4f, \"p999\": %.4f, \"max\": %.4f}",
                s.n, s.mean, s.sd, s.min, s.p50, s.p95, s.p99, s.p999, s.max);
  return buf;
}

struct NvmlSampler {
  std::vector<double> util, mem_util, dev_mem_mb, proc_mem_mb;
  std::thread th;
  std::atomic<bool> stop{false};
  bool ok = false;
  std::string err;

  void start(int period_ms) {
    nvmlReturn_t r = nvmlInit_v2();
    if (r != NVML_SUCCESS) { err = nvmlErrorString(r); return; }
    nvmlDevice_t dev;
    r = nvmlDeviceGetHandleByIndex_v2(0, &dev);
    if (r != NVML_SUCCESS) { err = nvmlErrorString(r); return; }
    ok = true;
    th = std::thread([this, dev, period_ms] {
      sched_param sp{};  // the sampler must not inherit SCHED_FIFO from `chrt -f`
      pthread_setschedparam(pthread_self(), SCHED_OTHER, &sp);
      const unsigned pid = static_cast<unsigned>(getpid());
      while (!stop) {
        nvmlUtilization_t u;
        if (nvmlDeviceGetUtilizationRates(dev, &u) == NVML_SUCCESS) {
          util.push_back(u.gpu);
          mem_util.push_back(u.memory);
        }
        nvmlMemory_t m;
        if (nvmlDeviceGetMemoryInfo(dev, &m) == NVML_SUCCESS) dev_mem_mb.push_back(static_cast<double>(m.used) / 1048576.0);
        unsigned n = 16;
        nvmlProcessInfo_t procs[16];
        if (nvmlDeviceGetComputeRunningProcesses(dev, &n, procs) == NVML_SUCCESS) {
          for (unsigned i = 0; i < n; ++i)
            if (procs[i].pid == pid) proc_mem_mb.push_back(static_cast<double>(procs[i].usedGpuMemory) / 1048576.0);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(period_ms));
      }
    });
  }
  void finish() {
    stop = true;
    if (th.joinable()) th.join();
    if (ok) nvmlShutdown();
  }
};

double vmean(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x;
  return v.empty() ? NAN : s / static_cast<double>(v.size());
}
double vmax(const std::vector<double>& v) {
  double m = v.empty() ? NAN : v[0];
  for (double x : v) m = std::max(m, x);
  return m;
}
std::string num(double x) {
  if (std::isnan(x)) return "null";
  char b[64];
  std::snprintf(b, sizeof(b), "%.4f", x);
  return b;
}

}  // namespace

int main(int argc, char** argv) {
  std::string engine, frame, out, stream_prio = "default", hog_prio = "least", cpus;
  double rate = 30.0, deadline_ms = -1;
  int iters = 2000, warmup = 100, fifo = 0, hog_blocks = 0, hog_inner = 20000;
  long timerslack_ns = -1;
  bool do_mlock = false, blocking_sync = false;
  std::vector<std::pair<std::string, std::string>> meta;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--engine") engine = val();
    else if (k == "--frame") frame = val();
    else if (k == "--out") out = val();
    else if (k == "--rate") rate = std::stod(val());
    else if (k == "--deadline-ms") deadline_ms = std::stod(val());
    else if (k == "--iters") iters = std::stoi(val());
    else if (k == "--warmup") warmup = std::stoi(val());
    else if (k == "--cpus") cpus = val();
    else if (k == "--fifo") fifo = std::stoi(val());
    else if (k == "--mlock") do_mlock = true;
    else if (k == "--timerslack-ns") timerslack_ns = std::stol(val());
    else if (k == "--blocking-sync") blocking_sync = true;
    else if (k == "--stream-prio") stream_prio = val();
    else if (k == "--inproc-hog") hog_blocks = std::stoi(val());
    else if (k == "--inproc-hog-prio") hog_prio = val();
    else if (k == "--hog-inner") hog_inner = std::stoi(val());
    else if (k == "--meta") {
      std::string kv = val();
      auto eq = kv.find('=');
      if (eq == std::string::npos) throw std::runtime_error("--meta expects key=value");
      meta.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
    } else {
      std::cerr << "unknown arg " << k << "\n";
      return 2;
    }
  }
  if (engine.empty() || frame.empty()) {
    std::cerr << "usage: latency_bench --engine E --frame DIR [options], see source header\n";
    return 2;
  }
  signal(SIGTERM, on_sig);
  signal(SIGINT, on_sig);
  const bool free_run = rate <= 0;
  const int64_t period_ns = free_run ? 0 : static_cast<int64_t>(std::llround(1e9 / rate));
  if (deadline_ms < 0) deadline_ms = free_run ? 0 : 1e3 / rate;

  // CPU affinity before any CUDA/TensorRT init so that every helper thread inherits it.
  std::string affinity_status = "inherited";
  if (!cpus.empty()) {
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : parse_cpus(cpus)) CPU_SET(c, &set);
    affinity_status = sched_setaffinity(0, sizeof(set), &set) == 0 ? "set:" + cpus : std::string("failed:") + strerror(errno);
  }
  if (blocking_sync) CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

  int prio_least = 0, prio_greatest = 0;
  CK(cudaDeviceGetStreamPriorityRange(&prio_least, &prio_greatest));
  const int sp = stream_prio == "high" ? prio_greatest : stream_prio == "low" ? prio_least : 0;
  obf::TrtRunner runner(engine, sp);
  cudaStream_t st = runner.stream();

  // Host buffers: pageable source frame (as a sensor driver would hand it over) + pinned staging / output buffers.
  struct Buf { const obf::TensorInfo* t; std::vector<char> src; void* pinned; };
  std::vector<Buf> ins, outs;
  size_t in_bytes = 0, out_bytes = 0;
  for (const auto& t : runner.tensors()) {
    Buf b{&t, {}, nullptr};
    CK(cudaHostAlloc(&b.pinned, t.bytes, cudaHostAllocDefault));
    if (t.is_input) {
      b.src = read_tensor(frame, t.name, t.bytes);
      in_bytes += t.bytes;
      ins.push_back(std::move(b));
    } else {
      out_bytes += t.bytes;
      outs.push_back(std::move(b));
    }
  }
  cudaEvent_t ev[4];
  for (auto& e : ev) CK(cudaEventCreate(&e));

  // Optional same-process GPU co-tenant on its own stream (tests whether stream priorities protect inference).
  std::thread hog;
  std::atomic<bool> hog_stop{false};
  std::atomic<long> hog_launches{0};
  float* hog_buf = nullptr;
  if (hog_blocks > 0) {
    CK(cudaMalloc(&hog_buf, sizeof(float) * hog_blocks * 256));
    CK(cudaMemset(hog_buf, 0, sizeof(float) * hog_blocks * 256));
    const int hp = hog_prio == "greatest" ? prio_greatest : prio_least;
    hog = std::thread([&, hp] {
      sched_param p{};
      pthread_setschedparam(pthread_self(), SCHED_OTHER, &p);
      cudaStream_t hs;
      cudaStreamCreateWithPriority(&hs, cudaStreamNonBlocking, hp);
      while (!hog_stop) {
        obf::rt::launch_hog(hs, hog_buf, hog_blocks, hog_inner);
        cudaStreamSynchronize(hs);
        ++hog_launches;
      }
      cudaStreamDestroy(hs);
    });
  }

  std::string mlock_status = "off";
  if (do_mlock) mlock_status = mlockall(MCL_CURRENT | MCL_FUTURE) == 0 ? "ok" : std::string("failed:") + strerror(errno);
  // Real-time policy for the loop thread (equivalent to `chrt -f` restricted to this thread).
  std::string fifo_status = "off";
  if (fifo > 0) {
    sched_param p{};
    p.sched_priority = fifo;
    const int rc = pthread_setschedparam(pthread_self(), SCHED_FIFO, &p);
    fifo_status = rc == 0 ? "ok:" + std::to_string(fifo) : std::string("failed:") + strerror(rc);
  }
  {
    int pol = 0;
    sched_param p{};
    pthread_getschedparam(pthread_self(), &pol, &p);
    if (pol == SCHED_FIFO && fifo == 0) fifo_status = "inherited:" + std::to_string(p.sched_priority);
  }

  // Timer slack (default 50 us for SCHED_OTHER threads) delays every clock_nanosleep wake-up by up to that amount.
  if (timerslack_ns >= 0) prctl(PR_SET_TIMERSLACK, timerslack_ns > 0 ? timerslack_ns : 1, 0, 0, 0);
  const long timerslack_eff = prctl(PR_GET_TIMERSLACK, 0, 0, 0, 0);

  NvmlSampler nvml;
  const int total = warmup + iters;
  std::vector<int64_t> rel(total), wake(total), done(total), skipped(total, 0);
  std::vector<double> pre_ms(total), h2d_ms(total), inf_ms(total), d2h_ms(total);
  rusage ru0{}, ru1{}, rt0{}, rt1{};
  int64_t t_meas0 = 0;
  double checksum = 0;

  nvml.start(50);  // started before the loop so nvmlInit does not perturb a measured iteration
  int64_t release = now_ns() + 50000000LL;  // first release 50 ms from now
  if (!free_run) sleep_until(release);
  int n_done = 0;
  for (int i = 0; i < total && !g_stop; ++i) {
    if (i == warmup) {
      getrusage(RUSAGE_SELF, &ru0);
      getrusage(RUSAGE_THREAD, &rt0);
      t_meas0 = now_ns();
    }
    const int64_t tw = now_ns();
    rel[i] = free_run ? tw : release;
    wake[i] = tw;
    for (auto& b : ins) std::memcpy(b.pinned, b.src.data(), b.t->bytes);  // host staging (pre-processing stand-in)
    const int64_t tp = now_ns();
    CK(cudaEventRecord(ev[0], st));
    for (auto& b : ins) CK(cudaMemcpyAsync(b.t->dev, b.pinned, b.t->bytes, cudaMemcpyHostToDevice, st));
    CK(cudaEventRecord(ev[1], st));
    runner.enqueue();
    CK(cudaEventRecord(ev[2], st));
    for (auto& b : outs) CK(cudaMemcpyAsync(b.pinned, b.t->dev, b.t->bytes, cudaMemcpyDeviceToHost, st));
    CK(cudaEventRecord(ev[3], st));
    CK(cudaStreamSynchronize(st));
    const int64_t td = now_ns();
    done[i] = td;
    pre_ms[i] = static_cast<double>(tp - tw) / 1e6;
    float a = 0, b = 0, c = 0;
    CK(cudaEventElapsedTime(&a, ev[0], ev[1]));
    CK(cudaEventElapsedTime(&b, ev[1], ev[2]));
    CK(cudaEventElapsedTime(&c, ev[2], ev[3]));
    h2d_ms[i] = a;
    inf_ms[i] = b;
    d2h_ms[i] = c;
    checksum += static_cast<const float*>(outs[0].pinned)[0];
    ++n_done;
    if (!free_run) {
      skipped[i] = obf::rt::advance_release(release, period_ns, now_ns());
      sleep_until(release);
    }
  }
  const int64_t t_meas1 = now_ns();
  getrusage(RUSAGE_SELF, &ru1);
  getrusage(RUSAGE_THREAD, &rt1);
  nvml.finish();
  hog_stop = true;
  if (hog.joinable()) hog.join();
  const double rss_end = rss_mb();

  if (n_done <= warmup) {
    std::cerr << "interrupted before measurement started\n";
    return 1;
  }
  const size_t n = static_cast<size_t>(n_done - warmup);
  auto slice = [&](const std::vector<double>& v) { return std::vector<double>(v.begin() + warmup, v.begin() + n_done); };
  std::vector<double> e2e(n), resp(n);
  std::vector<int64_t> relm(rel.begin() + warmup, rel.begin() + n_done), wakem(wake.begin() + warmup, wake.begin() + n_done);
  long skipped_total = 0;
  for (size_t j = 0; j < n; ++j) {
    const size_t i = j + warmup;
    e2e[j] = static_cast<double>(done[i] - wake[i]) / 1e6;
    resp[j] = static_cast<double>(done[i] - rel[i]) / 1e6;
    skipped_total += skipped[i];
  }
  const std::vector<double> late_us = obf::rt::wake_lateness_us(relm, wakem);
  const size_t misses = free_run ? 0 : obf::rt::count_over(resp, deadline_ms);
  const auto s_e2e = obf::rt::summarize(e2e), s_resp = obf::rt::summarize(resp), s_late = obf::rt::summarize(late_us);
  const auto s_pre = obf::rt::summarize(slice(pre_ms)), s_h2d = obf::rt::summarize(slice(h2d_ms));
  const auto s_inf = obf::rt::summarize(slice(inf_ms)), s_d2h = obf::rt::summarize(slice(d2h_ms));
  const double wall_s = static_cast<double>(t_meas1 - t_meas0) / 1e9;
  const double cpu_user = tv_s(ru1.ru_utime) - tv_s(ru0.ru_utime), cpu_sys = tv_s(ru1.ru_stime) - tv_s(ru0.ru_stime);

  std::printf("n=%zu rate=%.1fHz e2e p50=%.3f p99=%.3f p99.9=%.3f max=%.3f ms | wake-late sd=%.1f us max=%.1f us | "
              "misses=%zu skipped=%ld | cpu=%.1f%% nivcsw=%ld\n",
              n, rate, s_e2e.p50, s_e2e.p99, s_e2e.p999, s_e2e.max, s_late.sd, s_late.max, misses, skipped_total,
              100.0 * (cpu_user + cpu_sys) / wall_s, ru1.ru_nivcsw - ru0.ru_nivcsw);

  if (!out.empty()) {
    {
      std::ofstream csv(out + ".csv");
      csv << "iter,release_ns,wake_late_us,pre_ms,h2d_ms,infer_ms,d2h_ms,e2e_ms,response_ms,skipped_after\n";
      char line[256];
      for (size_t j = 0; j < n; ++j) {
        const size_t i = j + warmup;
        std::snprintf(line, sizeof(line), "%zu,%lld,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%lld\n", j,
                      static_cast<long long>(rel[i] - rel[warmup]), late_us[j], pre_ms[i], h2d_ms[i], inf_ms[i],
                      d2h_ms[i], e2e[j], resp[j], static_cast<long long>(skipped[i]));
        csv << line;
      }
    }
    utsname un{};
    uname(&un);
    int dev = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    std::ofstream js(out + ".json");
    js << "{\n  \"meta\": {";
    for (size_t k = 0; k < meta.size(); ++k)
      js << (k ? ", " : "") << "\"" << json_escape(meta[k].first) << "\": \"" << json_escape(meta[k].second) << "\"";
    js << "},\n";
    js << "  \"host\": {\"kernel\": \"" << un.release << "\", \"kernel_version\": \"" << json_escape(un.version)
       << "\", \"gpu\": \"" << json_escape(prop.name) << "\", \"nproc\": " << sysconf(_SC_NPROCESSORS_ONLN) << "},\n";
    js << "  \"config\": {\"engine\": \"" << json_escape(engine) << "\", \"rate_hz\": " << rate
       << ", \"deadline_ms\": " << num(deadline_ms) << ", \"iters\": " << n << ", \"warmup\": " << warmup
       << ", \"input_bytes\": " << in_bytes << ", \"output_bytes\": " << out_bytes << ", \"affinity\": \""
       << affinity_status << "\", \"fifo\": \"" << fifo_status << "\", \"mlock\": \"" << mlock_status << "\", \"timerslack_ns\": \"" << timerslack_eff
       << "\", \"stream_priority\": " << sp << ", \"stream_priority_range\": [" << prio_least << ", " << prio_greatest
       << "], \"blocking_sync\": " << (blocking_sync ? "true" : "false") << ", \"inproc_hog_blocks\": " << hog_blocks
       << ", \"inproc_hog_prio\": \"" << hog_prio << "\", \"inproc_hog_launches\": " << hog_launches.load() << "},\n";
    js << "  \"e2e_ms\": " << summary_json(s_e2e) << ",\n";
    js << "  \"response_ms\": " << summary_json(s_resp) << ",\n";
    js << "  \"wake_late_us\": " << summary_json(s_late) << ",\n";
    js << "  \"stages_ms\": {\"pre\": " << summary_json(s_pre) << ", \"h2d\": " << summary_json(s_h2d)
       << ", \"infer\": " << summary_json(s_inf) << ", \"d2h\": " << summary_json(s_d2h) << "},\n";
    js << "  \"deadline_misses\": " << misses << ",\n  \"skipped_releases\": " << skipped_total << ",\n";
    js << "  \"resources\": {\"wall_s\": " << num(wall_s) << ", \"cpu_user_s\": " << num(cpu_user)
       << ", \"cpu_sys_s\": " << num(cpu_sys) << ", \"cpu_pct_of_one_core\": " << num(100.0 * (cpu_user + cpu_sys) / wall_s)
       << ", \"proc_vol_csw\": " << ru1.ru_nvcsw - ru0.ru_nvcsw << ", \"proc_invol_csw\": " << ru1.ru_nivcsw - ru0.ru_nivcsw
       << ", \"loop_thread_vol_csw\": " << rt1.ru_nvcsw - rt0.ru_nvcsw
       << ", \"loop_thread_invol_csw\": " << rt1.ru_nivcsw - rt0.ru_nivcsw << ", \"maxrss_mb\": "
       << num(static_cast<double>(ru1.ru_maxrss) / 1024.0) << ", \"rss_end_mb\": " << num(rss_end)
       << ", \"nvml_ok\": " << (nvml.ok ? "true" : "false") << ", \"nvml_samples\": " << nvml.util.size()
       << ", \"gpu_util_mean_pct\": " << num(vmean(nvml.util)) << ", \"gpu_util_max_pct\": " << num(vmax(nvml.util))
       << ", \"gpu_mem_util_mean_pct\": " << num(vmean(nvml.mem_util)) << ", \"gpu_dev_mem_used_max_mb\": "
       << num(vmax(nvml.dev_mem_mb)) << ", \"gpu_proc_mem_max_mb\": " << num(vmax(nvml.proc_mem_mb)) << "},\n";
    js << "  \"checksum\": " << num(checksum) << "\n}\n";
  }
  for (auto& b : ins) cudaFreeHost(b.pinned);
  for (auto& b : outs) cudaFreeHost(b.pinned);
  if (hog_buf) cudaFree(hog_buf);
  return 0;
}
