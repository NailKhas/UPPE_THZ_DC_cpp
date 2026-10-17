#pragma once

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>

namespace uppe {

using cplx = std::complex<double>;

inline int max_threads() { return omp_get_max_threads(); }
inline int thread_id() { return omp_get_thread_num(); }

inline double sq(double x) { return x * x; }

inline double pow8(double x) {
  const double x2 = x * x;
  const double x4 = x2 * x2;
  return x4 * x4;
}

inline double pow16(double x) {
  const double x8 = pow8(x);
  return x8 * x8;
}

inline double pow20(double x) {
  const double x2 = x * x;
  const double x4 = x2 * x2;
  const double x8 = x4 * x4;
  const double x16 = x8 * x8;
  return x16 * x4;
}

inline double abs2(const cplx& z) { return z.real() * z.real() + z.imag() * z.imag(); }

// Textbook complex product, as Fortran compiles it (no C99 Annex G Inf/NaN recovery).
inline cplx cmul(const cplx& a, const cplx& b) {
  return cplx(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

// Angular filter exp(-(kr/krmax)^20) of LinThread, KdivKZ and FirstStep.
// krmax == 0 is the zero-frequency bin, where Fortran evaluates exp(-Inf) = 0.
// For x >= 2 the exponent is below -2^20 and exp underflows to exactly 0.
inline double kr_filter(double kr, double krmax) {
  if (!(krmax > 0.0)) return 0.0;
  const double x = kr / krmax;
  if (x >= 2.0) return 0.0;
  return std::exp(-pow20(x));
}

// Zero-initialized, 64-byte aligned, move-only buffer for the large field arrays.
template <class T>
class AlignedArray {
  static_assert(std::is_trivially_copyable<T>::value, "AlignedArray holds trivially copyable types only");

 public:
  AlignedArray() = default;
  explicit AlignedArray(std::size_t n) { resize(n); }
  ~AlignedArray() { release(); }

  AlignedArray(const AlignedArray&) = delete;
  AlignedArray& operator=(const AlignedArray&) = delete;

  AlignedArray(AlignedArray&& o) noexcept : raw_(o.raw_), p_(o.p_), n_(o.n_) {
    o.raw_ = nullptr;
    o.p_ = nullptr;
    o.n_ = 0;
  }
  AlignedArray& operator=(AlignedArray&& o) noexcept {
    if (this != &o) {
      release();
      raw_ = o.raw_;
      p_ = o.p_;
      n_ = o.n_;
      o.raw_ = nullptr;
      o.p_ = nullptr;
      o.n_ = 0;
    }
    return *this;
  }

  // Reallocates; the new contents are zero (all-bits-zero is 0.0 for double and complex).
  void resize(std::size_t n) {
    release();
    if (n == 0) return;
    raw_ = std::malloc(n * sizeof(T) + kAlign);
    if (!raw_) throw std::bad_alloc();
    p_ = reinterpret_cast<T*>((reinterpret_cast<std::uintptr_t>(raw_) + kAlign - 1) &
                              ~static_cast<std::uintptr_t>(kAlign - 1));
    n_ = n;
    fill_zero();
  }

  void release() {
    std::free(raw_);
    raw_ = nullptr;
    p_ = nullptr;
    n_ = 0;
  }

  void fill_zero() {
    const std::size_t bytes = n_ * sizeof(T);
    if (bytes == 0) return;
    char* base = reinterpret_cast<char*>(p_);
    const std::size_t chunk = std::size_t(1) << 22;
    const int nchunks = static_cast<int>((bytes + chunk - 1) / chunk);
#pragma omp parallel for schedule(static)
    for (int c = 0; c < nchunks; ++c) {
      const std::size_t off = static_cast<std::size_t>(c) * chunk;
      std::memset(base + off, 0, std::min(chunk, bytes - off));
    }
  }

  T* data() { return p_; }
  const T* data() const { return p_; }
  std::size_t size() const { return n_; }
  T& operator[](std::size_t i) { return p_[i]; }
  const T& operator[](std::size_t i) const { return p_[i]; }

 private:
  static constexpr std::size_t kAlign = 64;
  void* raw_ = nullptr;
  T* p_ = nullptr;
  std::size_t n_ = 0;
};

}  // namespace uppe
