#include "rt_stats.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace obf::rt {

double percentile_sorted(const std::vector<double>& s, double q) {
  if (s.empty()) throw std::invalid_argument("percentile of empty sample");
  if (!(q >= 0.0 && q <= 1.0)) throw std::invalid_argument("percentile q must be in [0, 1]");
  const double pos = q * static_cast<double>(s.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = std::min(lo + 1, s.size() - 1);
  const double frac = pos - static_cast<double>(lo);
  return s[lo] + (s[hi] - s[lo]) * frac;
}

double percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return percentile_sorted(v, q);
}

double stddev(const std::vector<double>& v) {
  if (v.size() < 2) return 0.0;
  const double m = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
  double ss = 0.0;
  for (double x : v) ss += (x - m) * (x - m);
  return std::sqrt(ss / static_cast<double>(v.size() - 1));
}

Summary summarize(const std::vector<double>& v) {
  Summary r;
  if (v.empty()) return r;
  std::vector<double> s(v);
  std::sort(s.begin(), s.end());
  r.n = s.size();
  r.mean = std::accumulate(s.begin(), s.end(), 0.0) / static_cast<double>(s.size());
  r.sd = stddev(s);
  r.min = s.front();
  r.max = s.back();
  r.p50 = percentile_sorted(s, 0.50);
  r.p95 = percentile_sorted(s, 0.95);
  r.p99 = percentile_sorted(s, 0.99);
  r.p999 = percentile_sorted(s, 0.999);
  return r;
}

std::vector<double> wake_lateness_us(const std::vector<int64_t>& planned, const std::vector<int64_t>& actual) {
  if (planned.size() != actual.size()) throw std::invalid_argument("planned/actual size mismatch");
  std::vector<double> out(planned.size());
  for (size_t i = 0; i < planned.size(); ++i) out[i] = static_cast<double>(actual[i] - planned[i]) / 1e3;
  return out;
}

size_t count_over(const std::vector<double>& v, double limit) {
  return static_cast<size_t>(std::count_if(v.begin(), v.end(), [limit](double x) { return x > limit; }));
}

int64_t advance_release(int64_t& release_ns, int64_t period_ns, int64_t now_ns) {
  if (period_ns <= 0) throw std::invalid_argument("period must be > 0");
  release_ns += period_ns;
  if (release_ns >= now_ns) return 0;
  // Frame-drop policy: releases strictly before `now` are stale (a newer sensor frame exists) and are skipped; the
  // next job starts at the first grid point >= now.
  const int64_t behind = (now_ns - release_ns + period_ns - 1) / period_ns;
  release_ns += behind * period_ns;
  return behind;
}

}  // namespace obf::rt
