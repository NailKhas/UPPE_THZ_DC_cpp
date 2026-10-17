#include "fft.hpp"

#include <mutex>
#include <stdexcept>

#if defined(UPPE_USE_MKL)
#include <fftw/fftw3.h>
#else
#include <fftw3.h>
#endif

namespace uppe {

namespace {
std::mutex& planner_mutex() {
  static std::mutex m;
  return m;
}
}  // namespace

RealFFT::RealFFT(int n) : n_(n) {
  if (n < 2 || n % 2 != 0) throw std::invalid_argument("RealFFT: size must be even");
  std::lock_guard<std::mutex> lock(planner_mutex());
  time_ = static_cast<double*>(fftw_malloc(sizeof(double) * static_cast<std::size_t>(n)));
  void* s = fftw_malloc(sizeof(fftw_complex) * static_cast<std::size_t>(n / 2 + 1));
  spec_ = static_cast<cplx*>(s);
  if (!time_ || !spec_) {
    fftw_free(time_);
    fftw_free(s);
    throw std::bad_alloc();
  }
  fftw_complex* sc = static_cast<fftw_complex*>(s);
  fftw_plan f = fftw_plan_dft_r2c_1d(n, time_, sc, FFTW_MEASURE);
  fftw_plan b = fftw_plan_dft_c2r_1d(n, sc, time_, FFTW_MEASURE);
  if (!f || !b) {
    if (f) fftw_destroy_plan(f);
    if (b) fftw_destroy_plan(b);
    fftw_free(time_);
    fftw_free(s);
    throw std::runtime_error("RealFFT: FFTW planning failed");
  }
  fwd_ = f;
  bwd_ = b;
  for (int i = 0; i < n; ++i) time_[i] = 0.0;
  for (int q = 0; q <= n / 2; ++q) spec_[q] = cplx(0.0, 0.0);
}

RealFFT::~RealFFT() {
  std::lock_guard<std::mutex> lock(planner_mutex());
  fftw_destroy_plan(static_cast<fftw_plan>(fwd_));
  fftw_destroy_plan(static_cast<fftw_plan>(bwd_));
  fftw_free(time_);
  fftw_free(spec_);
}

void RealFFT::forward() { fftw_execute(static_cast<fftw_plan>(fwd_)); }

void RealFFT::backward() { fftw_execute(static_cast<fftw_plan>(bwd_)); }

FftPool make_fft_pool(int n, int count) {
  FftPool pool;
  pool.reserve(count);
  for (int i = 0; i < count; ++i) pool.push_back(std::make_unique<RealFFT>(n));
  return pool;
}

void centered_forward(RealFFT& fft, cplx* half, int nHalf) {
  if (nHalf < 0 || nHalf > fft.size() / 2) throw std::invalid_argument("centered_forward: bad nHalf");
  fft.forward();
  for (int q = 0; q < nHalf; ++q) half[q] = centered_bin(fft, q);
}

void centered_backward(RealFFT& fft, const cplx* half, int nHalf) {
  const int n = fft.size();
  const int h = n / 2;
  if (nHalf < 0 || nHalf > h) throw std::invalid_argument("centered_backward: bad nHalf");
  cplx* s = fft.spec();
  for (int q = 0; q < nHalf; ++q) s[q] = (q & 1) ? -half[q] : half[q];
  for (int q = nHalf; q <= h; ++q) s[q] = cplx(0.0, 0.0);
  fft.backward();
  double* t = fft.time();
  const double dn = static_cast<double>(n);
  for (int i = 0; i < n; ++i) t[i] /= dn;
}

}  // namespace uppe
