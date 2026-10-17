#pragma once

#include <math.h>

namespace uppe {

// Bessel functions of the first kind from the C runtime (replacing Intel's dbesj0/dbesj1).
inline double bessel_j0(double x) {
#if defined(_WIN32)
  return ::_j0(x);
#else
  return ::j0(x);
#endif
}

inline double bessel_j1(double x) {
#if defined(_WIN32)
  return ::_j1(x);
#else
  return ::j1(x);
#endif
}

}  // namespace uppe
