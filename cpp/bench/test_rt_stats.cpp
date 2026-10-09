// ctest target for rt_stats (plain asserts, no GoogleTest download needed). Expected values cross-checked against
// numpy.percentile(..., method="linear") and numpy.std(..., ddof=1).
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "rt_stats.hpp"

using namespace obf::rt;

static int g_fail = 0, g_pass = 0;
#define EXPECT_NEAR(a, b, tol)                                                                    \
  do {                                                                                            \
    double _a = (a), _b = (b);                                                                    \
    if (std::fabs(_a - _b) > (tol)) {                                                             \
      std::printf("FAIL %s:%d %s = %.9g, expected %.9g\n", __FILE__, __LINE__, #a, _a, _b);       \
      ++g_fail;                                                                                   \
    } else {                                                                                      \
      ++g_pass;                                                                                   \
    }                                                                                             \
  } while (0)
#define EXPECT_THROW(stmt)                                                \
  do {                                                                    \
    bool _t = false;                                                      \
    try { stmt; } catch (const std::invalid_argument&) { _t = true; }     \
    if (!_t) { std::printf("FAIL %s:%d no throw\n", __FILE__, __LINE__); ++g_fail; } else { ++g_pass; } \
  } while (0)

static void test_percentile_linear() {
  std::vector<double> v{1, 2, 3, 4};
  EXPECT_NEAR(percentile(v, 0.0), 1.0, 1e-12);
  EXPECT_NEAR(percentile(v, 1.0), 4.0, 1e-12);
  EXPECT_NEAR(percentile(v, 0.5), 2.5, 1e-12);   // np.percentile([1,2,3,4], 50) == 2.5
  EXPECT_NEAR(percentile(v, 0.99), 3.97, 1e-12); // np.percentile([1,2,3,4], 99) == 3.97
  std::vector<double> unsorted{10, 1, 7, 3, 5};
  EXPECT_NEAR(percentile(unsorted, 0.5), 5.0, 1e-12);
  EXPECT_NEAR(percentile(unsorted, 0.25), 3.0, 1e-12);
  std::vector<double> one{42};
  EXPECT_NEAR(percentile(one, 0.999), 42.0, 1e-12);
}

static void test_percentile_1_to_1000() {
  std::vector<double> v;
  for (int i = 1; i <= 1000; ++i) v.push_back(i);
  EXPECT_NEAR(percentile(v, 0.999), 999.001, 1e-9);  // numpy: 999.001
  EXPECT_NEAR(percentile(v, 0.95), 950.05, 1e-9);
  Summary s = summarize(v);
  EXPECT_NEAR(static_cast<double>(s.n), 1000, 0);
  EXPECT_NEAR(s.mean, 500.5, 1e-9);
  EXPECT_NEAR(s.sd, 288.8194360957494, 1e-9);  // np.std(1..1000, ddof=1)
  EXPECT_NEAR(s.p50, 500.5, 1e-9);
  EXPECT_NEAR(s.p99, 990.01, 1e-9);
  EXPECT_NEAR(s.min, 1, 0);
  EXPECT_NEAR(s.max, 1000, 0);
}

static void test_errors() {
  EXPECT_THROW(percentile({}, 0.5));
  EXPECT_THROW(percentile({1.0}, 1.5));
  EXPECT_THROW(percentile({1.0}, -0.1));
  EXPECT_THROW(wake_lateness_us({1, 2}, {1}));
  int64_t r = 0;
  EXPECT_THROW(advance_release(r, 0, 0));
  EXPECT_NEAR(stddev({5.0}), 0.0, 0);
  EXPECT_NEAR(static_cast<double>(summarize({}).n), 0, 0);
}

static void test_jitter_and_misses() {
  std::vector<int64_t> planned{1000000, 2000000, 3000000};
  std::vector<int64_t> actual{1000050, 2000000, 3001500};
  auto w = wake_lateness_us(planned, actual);
  EXPECT_NEAR(w[0], 0.05, 1e-12);
  EXPECT_NEAR(w[1], 0.0, 1e-12);
  EXPECT_NEAR(w[2], 1.5, 1e-12);
  EXPECT_NEAR(stddev(w), 0.8519585279, 1e-9);  // np.std([0.05, 0, 1.5], ddof=1)
  std::vector<double> resp{10.0, 33.3, 33.4, 50.0};
  EXPECT_NEAR(static_cast<double>(count_over(resp, 33.3)), 2, 0);  // strictly greater
}

static void test_advance_release() {
  const int64_t P = 1000;
  int64_t rel = 0;
  EXPECT_NEAR(static_cast<double>(advance_release(rel, P, 500)), 0, 0);  // finished inside the period
  EXPECT_NEAR(static_cast<double>(rel), 1000, 0);
  rel = 0;
  EXPECT_NEAR(static_cast<double>(advance_release(rel, P, 1000)), 0, 0);  // finished exactly at next release
  EXPECT_NEAR(static_cast<double>(rel), 1000, 0);
  rel = 0;
  EXPECT_NEAR(static_cast<double>(advance_release(rel, P, 3000)), 2, 0);  // 1000, 2000 lost, start at 3000
  EXPECT_NEAR(static_cast<double>(rel), 3000, 0);
  rel = 0;
  EXPECT_NEAR(static_cast<double>(advance_release(rel, P, 3500)), 3, 0);  // releases 1000, 2000, 3000 lost
  EXPECT_NEAR(static_cast<double>(rel), 4000, 0);
}

int main() {
  test_percentile_linear();
  test_percentile_1_to_1000();
  test_errors();
  test_jitter_and_misses();
  test_advance_release();
  std::printf("rt_stats tests: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
