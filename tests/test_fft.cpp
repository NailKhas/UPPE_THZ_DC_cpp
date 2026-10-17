#include <vector>

#include "check.hpp"
#include "fft.hpp"
#include "reference.hpp"

using namespace uppe;

int main() {
  test::init();
  test::Rng rng;

  // Forward: real signal -> non-negative half of the Fortran-swapped spectrum.
  {
    const int N = 64;
    RealFFT f(N);
    ref::Vec x(N);
    for (int n = 0; n < N; ++n) {
      const double v = rng.uniform(-1.0, 1.0);
      f.time()[n] = v;
      x[n] = v;
    }
    std::vector<cplx> half(N / 2);
    centered_forward(f, half.data(), N / 2);
    const ref::Vec R = ref::fortran_forward(x);
    CHECK(test::rel_err(half.data(), R.data() + N / 2, N / 2) < 1e-13);
    for (int q = 0; q < N / 2; ++q) CHECK_CLOSE(centered_bin(f, q).real(), R[N / 2 + q].real(), 1e-12);

    // Partial output.
    std::vector<cplx> part(5);
    for (int n = 0; n < N; ++n) f.time()[n] = x[n].real();
    centered_forward(f, part.data(), 5);
    CHECK(test::rel_err(part.data(), R.data() + N / 2, 5) < 1e-13);
  }

  // Backward: half spectrum -> real signal, divided by N.
  {
    const int N = 64;
    RealFFT f(N);
    std::vector<cplx> half(N / 2);
    for (int q = 1; q < N / 2; ++q) half[q] = rng.cuniform(1.0);
    half[0] = cplx(rng.uniform(-1.0, 1.0), 0.0);
    centered_backward(f, half.data(), N / 2);
    const ref::Vec R = ref::fortran_backward(ref::full_from_half(half.data(), N));
    double maxIm = 0.0;
    std::vector<double> re(N);
    for (int n = 0; n < N; ++n) {
      re[n] = R[n].real();
      maxIm = std::max(maxIm, std::fabs(R[n].imag()));
    }
    CHECK(maxIm < 1e-15);
    CHECK(test::rel_err(f.time(), re.data(), N) < 1e-13);
  }

  // Backward with embedding (JefThread_THz): ntT-point THz spectrum centred in an N-point array.
  {
    const int N = 512, ntT = 32;
    RealFFT f(N);
    std::vector<cplx> half(ntT / 2);
    for (int q = 1; q < ntT / 2; ++q) half[q] = rng.cuniform(1.0);
    centered_backward(f, half.data(), ntT / 2);
    const ref::Vec small = ref::full_from_half(half.data(), ntT);
    ref::Vec E1(N, cplx(0.0, 0.0));
    for (int i = 0; i < ntT; ++i) E1[N / 2 - ntT / 2 + i] = small[i];
    const ref::Vec R = ref::fortran_backward(E1);
    std::vector<double> re(N);
    for (int n = 0; n < N; ++n) re[n] = R[n].real();
    CHECK(test::rel_err(f.time(), re.data(), N) < 1e-13);

    // Round trip.
    std::vector<cplx> back(ntT / 2);
    centered_forward(f, back.data(), ntT / 2);
    CHECK(test::rel_err(back.data(), half.data(), ntT / 2) < 1e-13);
  }

  // Pool construction.
  {
    FftPool pool = make_fft_pool(128, 3);
    CHECK(pool.size() == 3);
    for (const auto& p : pool) CHECK(p && p->size() == 128);
  }

  return test::finish("test_fft");
}
