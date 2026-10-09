#pragma once
// Pure C++17 statistics for the real-time latency benchmark (no CUDA/TensorRT dependency, unit-tested in
// test_rt_stats.cpp). Percentiles use linear interpolation between closest ranks (same as numpy's default
// "linear" method) so the C++ summary and the Python report agree.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace obf::rt {

struct Summary {
  size_t n = 0;
  double mean = 0, sd = 0, min = 0, p50 = 0, p95 = 0, p99 = 0, p999 = 0, max = 0;
};

// q in [0, 1]. Throws std::invalid_argument on empty input or q outside [0, 1].
double percentile(std::vector<double> v, double q);
double percentile_sorted(const std::vector<double>& sorted, double q);
// Sample standard deviation (n - 1 denominator); 0 for n < 2.
double stddev(const std::vector<double>& v);
Summary summarize(const std::vector<double>& v);

// Release-jitter of a periodic task: wake lateness = actual wake time - planned release time (ns -> us).
std::vector<double> wake_lateness_us(const std::vector<int64_t>& planned_ns, const std::vector<int64_t>& actual_ns);

// Number of samples strictly greater than the limit (deadline misses when v is the response time).
size_t count_over(const std::vector<double>& v, double limit);

// Advance an absolute release time past `now_ns` on a fixed period grid. Returns the number of releases that were
// skipped because the previous job overran them (0 when the next release is still in the future).
int64_t advance_release(int64_t& release_ns, int64_t period_ns, int64_t now_ns);

}  // namespace obf::rt
