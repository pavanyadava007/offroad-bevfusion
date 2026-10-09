// gpu_hog: separate-process GPU co-tenant. Launches hog kernels back to back (or with a duty cycle) until SIGTERM/SIGINT
// or --seconds elapses, then prints the achieved busy time.
//   gpu_hog [--blocks 4640] [--inner 20000] [--duty 1.0] [--priority least|greatest] [--seconds 0]
#include <cuda_runtime.h>
#include <signal.h>
#include <time.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include "hog_kernel.cuh"

static std::atomic<bool> g_stop{false};
static void on_sig(int) { g_stop = true; }

int main(int argc, char** argv) {
  int blocks = 58 * 80, inner = 20000;
  double duty = 1.0, seconds = 0;
  std::string prio = "least";
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "--blocks") blocks = std::atoi(argv[i + 1]);
    else if (k == "--inner") inner = std::atoi(argv[i + 1]);
    else if (k == "--duty") duty = std::atof(argv[i + 1]);
    else if (k == "--priority") prio = argv[i + 1];
    else if (k == "--seconds") seconds = std::atof(argv[i + 1]);
    else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
  }
  signal(SIGTERM, on_sig);
  signal(SIGINT, on_sig);
  int lo = 0, hi = 0;
  cudaDeviceGetStreamPriorityRange(&lo, &hi);  // lo = least (numerically largest), hi = greatest
  cudaStream_t s;
  cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, prio == "greatest" ? hi : lo);
  float* buf = nullptr;
  cudaMalloc(&buf, sizeof(float) * blocks * 256);
  cudaMemset(buf, 0, sizeof(float) * blocks * 256);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  const auto t_start = std::chrono::steady_clock::now();
  double busy_ms = 0;
  long launches = 0;
  while (!g_stop) {
    cudaEventRecord(e0, s);
    if (obf::rt::launch_hog(s, buf, blocks, inner) != cudaSuccess) { std::fprintf(stderr, "launch failed\n"); return 1; }
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    busy_ms += ms;
    ++launches;
    if (duty < 1.0 && duty > 0.0) {  // idle so that busy / (busy + idle) ~= duty
      std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(ms * 1e3 * (1.0 - duty) / duty)));
    }
    if (seconds > 0 &&
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() > seconds) break;
  }
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  std::printf("gpu_hog launches=%ld mean_kernel_ms=%.3f busy_fraction=%.3f wall_s=%.1f\n", launches,
              launches ? busy_ms / launches : 0.0, busy_ms / 1e3 / wall, wall);
  cudaFree(buf);
  return 0;
}
