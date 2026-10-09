// cpu_hog: CPU / memory-bandwidth interference generator (stand-in for stress-ng, which is not installed).
//   cpu_hog --mode cpu|membw [--threads N] [--mb 256] [--seconds 0]
// cpu:   N threads spinning on dependent integer/float arithmetic (keeps every allowed core 100 % busy).
// membw: N threads memcpy-ing between two private buffers of --mb MiB each (streams through DRAM, saturates the
//        memory controllers and the shared L3). Restrict placement from outside with taskset / cpusets.
// Runs until SIGTERM/SIGINT (or --seconds), then prints the achieved operation rate.
#include <signal.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static std::atomic<bool> g_stop{false};
static void on_sig(int) { g_stop = true; }

int main(int argc, char** argv) {
  std::string mode = "cpu";
  int threads = static_cast<int>(std::thread::hardware_concurrency());
  size_t mb = 256;
  double seconds = 0;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "--mode") mode = argv[i + 1];
    else if (k == "--threads") threads = std::atoi(argv[i + 1]);
    else if (k == "--mb") mb = static_cast<size_t>(std::atol(argv[i + 1]));
    else if (k == "--seconds") seconds = std::atof(argv[i + 1]);
    else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
  }
  if (mode != "cpu" && mode != "membw") { std::fprintf(stderr, "mode must be cpu or membw\n"); return 2; }
  signal(SIGTERM, on_sig);
  signal(SIGINT, on_sig);
  std::vector<double> work(threads, 0.0);
  std::vector<std::thread> ts;
  const auto t0 = std::chrono::steady_clock::now();
  for (int t = 0; t < threads; ++t) {
    ts.emplace_back([&, t] {
      if (mode == "cpu") {
        volatile double x = 1.0 + t;
        double n = 0;
        while (!g_stop) {
          for (int k = 0; k < 100000; ++k) x = x * 1.0000001 + 1e-9;
          n += 1e5;
        }
        work[t] = n;
      } else {
        const size_t bytes = mb << 20;
        std::vector<char> a(bytes, 1), b(bytes, 2);
        double moved = 0;
        while (!g_stop) {
          std::memcpy(b.data(), a.data(), bytes);
          std::memcpy(a.data(), b.data(), bytes);
          moved += 4.0 * static_cast<double>(bytes);  // each memcpy reads + writes `bytes`
        }
        work[t] = moved;
      }
    });
  }
  if (seconds > 0) {
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    g_stop = true;
  }
  for (auto& th : ts) th.join();
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  double total = 0;
  for (double w : work) total += w;
  if (mode == "cpu") std::printf("cpu_hog mode=cpu threads=%d Mops_per_s=%.1f wall_s=%.1f\n", threads, total / wall / 1e6, wall);
  else std::printf("cpu_hog mode=membw threads=%d GB_per_s=%.1f wall_s=%.1f\n", threads, total / wall / 1e9, wall);
  return 0;
}
