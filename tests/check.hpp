#pragma once

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "blas.hpp"
#include "common.hpp"

namespace test {

inline int& failures() {
  static int f = 0;
  return f;
}

#define CHECK(cond)                                                              \
  do {                                                                           \
    if (!(cond)) {                                                               \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      ++test::failures();                                                        \
    }                                                                            \
  } while (0)

#define CHECK_CLOSE(a, b, tol)                                                                        \
  do {                                                                                                \
    const double a_ = (a), b_ = (b), t_ = (tol);                                                      \
    if (!(std::fabs(a_ - b_) <= t_)) {                                                                \
      std::fprintf(stderr, "%s:%d: %s = %.17g, %s = %.17g, |diff| = %.3g > %.3g\n", __FILE__, __LINE__, #a, \
                   a_, #b, b_, std::fabs(a_ - b_), t_);                                               \
      ++test::failures();                                                                             \
    }                                                                                                 \
  } while (0)

inline int finish(const char* name) {
  if (failures() > 0) {
    std::fprintf(stderr, "%s: %d failure(s)\n", name, failures());
    return 1;
  }
  std::printf("%s: OK\n", name);
  return 0;
}

inline void init() { uppe::blas_init(); }

// max |a - b| / max |b| over two complex sequences.
inline double rel_err(const uppe::cplx* a, const uppe::cplx* b, std::size_t n) {
  double dmax = 0.0, bmax = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    dmax = std::max(dmax, std::abs(a[i] - b[i]));
    bmax = std::max(bmax, std::abs(b[i]));
  }
  return bmax > 0.0 ? dmax / bmax : dmax;
}

inline double rel_err(const double* a, const double* b, std::size_t n) {
  double dmax = 0.0, bmax = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    dmax = std::max(dmax, std::fabs(a[i] - b[i]));
    bmax = std::max(bmax, std::fabs(b[i]));
  }
  return bmax > 0.0 ? dmax / bmax : dmax;
}

struct Rng {
  std::mt19937_64 gen{12345};
  double uniform(double lo, double hi) { return std::uniform_real_distribution<double>(lo, hi)(gen); }
  uppe::cplx cuniform(double amp) { return uppe::cplx(uniform(-amp, amp), uniform(-amp, amp)); }
};

}  // namespace test
