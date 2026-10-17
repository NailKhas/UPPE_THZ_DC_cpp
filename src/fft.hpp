#pragma once

#include <memory>
#include <vector>

#include "common.hpp"

namespace uppe {

// Real-to-complex / complex-to-real FFT of size n with its own aligned buffers and plans.
// Construction and destruction are serialized internally (FFTW planning is not thread-safe);
// forward()/backward() of different objects may run concurrently.
class RealFFT {
 public:
  explicit RealFFT(int n);
  ~RealFFT();
  RealFFT(const RealFFT&) = delete;
  RealFFT& operator=(const RealFFT&) = delete;

  int size() const { return n_; }
  double* time() { return time_; }
  const double* time() const { return time_; }
  cplx* spec() { return spec_; }  // n/2+1 bins
  const cplx* spec() const { return spec_; }

  void forward();   // time() -> spec(), exp(-i...) sign as DftiComputeForward
  void backward();  // spec() -> time(), unnormalized; spec() is overwritten

 private:
  int n_ = 0;
  double* time_ = nullptr;
  cplx* spec_ = nullptr;
  void* fwd_ = nullptr;
  void* bwd_ = nullptr;
};

using FftPool = std::vector<std::unique_ptr<RealFFT>>;
FftPool make_fft_pool(int n, int count);

// The Fortran code swaps array halves before and after every DFT (t = 0 and w = 0 at index n/2).
// For even n that is the same as multiplying the standard spectrum by (-1)^q.
// "half" holds the non-negative frequencies q = 0..n/2-1, i.e. Fortran indices n/2..n-1;
// the Nyquist bin (Fortran index 0) is always zero in this code.

// Spectrum of the signal in fft.time(): half[q] for q < nHalf (nHalf <= n/2).
void centered_forward(RealFFT& fft, cplx* half, int nHalf);

// Real time signal into fft.time() from half[0..nHalf), higher bins zero; divided by n
// like FurThread (Origin/dble(nt)).
void centered_backward(RealFFT& fft, const cplx* half, int nHalf);

inline cplx centered_bin(const RealFFT& fft, int q) {
  const cplx s = fft.spec()[q];
  return (q & 1) ? -s : s;
}

}  // namespace uppe
