#include "hankel.hpp"

#include <stdexcept>

#include "bessel.hpp"
#include "blas.hpp"

namespace uppe {

HankelMatrix make_hankel(const std::vector<double>& roots, int nR) {
  if (nR < 3 || static_cast<int>(roots.size()) < nR) throw std::invalid_argument("make_hankel: not enough roots");
  HankelMatrix H;
  H.n = nR - 1;
  const int n = H.n;
  H.Y.resize(static_cast<std::size_t>(n) * n);
  const double jN = roots[nR - 1];
  std::vector<double> j1sq(n);
  for (int k = 0; k < n; ++k) {
    const double b = bessel_j1(roots[k]);
    j1sq[k] = b * b;
  }
#pragma omp parallel for schedule(dynamic, 16)
  for (int k = 0; k < n; ++k) {
    double* row = H.Y.data() + static_cast<std::size_t>(k) * n;
    const double jk = roots[k];
    for (int i = 0; i < n; ++i) row[i] = 2.0 * bessel_j0(jk * roots[i] / jN) / j1sq[k] / jN;
  }
  return H;
}

void hankel_spectral(const HankelMatrix& H, const HankelInput& in, cplx* out, int M, int nq, const int* sumCut,
                     const int* outCut) {
  const int n = H.n;
  if (!in.a || in.rows < 1 || in.rows > n || M < 1 || M > n || nq < 1)
    throw std::invalid_argument("hankel_spectral: bad dimensions");
  if (out == in.a || (in.b && out == in.b)) throw std::invalid_argument("hankel_spectral: in-place call");

  const int nthreads = max_threads();
  int bw = 64;
  while (bw > 8 && (nq + bw - 1) / bw < 2 * nthreads) bw /= 2;
  const int nblocks = (nq + bw - 1) / bw;
  double* outd = reinterpret_cast<double*>(out);
  const int ldo = 2 * nq;
  const std::size_t snq = static_cast<std::size_t>(nq);

#pragma omp parallel
  {
    std::vector<double> scratch(static_cast<std::size_t>(in.rows) * 2 * bw);
#pragma omp for schedule(dynamic, 1)
    for (int blk = 0; blk < nblocks; ++blk) {
      const int q0 = blk * bw;
      const int w = std::min(bw, nq - q0);
      int K = in.rows;
      if (sumCut) {
        int m = 0;
        for (int q = q0; q < q0 + w; ++q) m = std::max(m, sumCut[q]);
        K = std::min(K, m);
      }
      int Mb = M;
      if (outCut) {
        int m = 0;
        for (int q = q0; q < q0 + w; ++q) m = std::max(m, outCut[q]);
        Mb = std::min(M, m);
      }
      const int ld = 2 * w;
      for (int k = 0; k < K; ++k) {
        const cplx* ra = in.a + static_cast<std::size_t>(k) * snq + q0;
        double* s = scratch.data() + static_cast<std::size_t>(k) * ld;
        if (in.b) {
          const cplx* rb = in.b + static_cast<std::size_t>(k) * snq + q0;
          for (int j = 0; j < w; ++j) {
            const cplx v = ra[j] + in.beta * rb[j];
            s[2 * j] = v.real();
            s[2 * j + 1] = v.imag();
          }
        } else {
          for (int j = 0; j < w; ++j) {
            s[2 * j] = ra[j].real();
            s[2 * j + 1] = ra[j].imag();
          }
        }
      }
      const int computed = K > 0 ? Mb : 0;
      if (computed > 0)
        dgemm_tn(computed, ld, K, H.Y.data(), n, scratch.data(), ld, outd + 2 * static_cast<std::size_t>(q0), ldo);
      for (int i = computed; i < M; ++i) {
        cplx* r = out + static_cast<std::size_t>(i) * snq + q0;
        std::fill(r, r + w, cplx(0.0, 0.0));
      }
    }
  }
}

void hankel_dense(const HankelMatrix& H, const double* in, double* out, int ncols) {
  const int n = H.n;
  if (ncols < 1 || in == out) throw std::invalid_argument("hankel_dense: bad arguments");
  const int nthreads = max_threads();
  int rows = (n + 4 * nthreads - 1) / (4 * nthreads);
  rows = std::max(rows, 64);
  const int ntasks = (n + rows - 1) / rows;
#pragma omp parallel for schedule(dynamic, 1)
  for (int t = 0; t < ntasks; ++t) {
    const int i0 = t * rows;
    const int m = std::min(rows, n - i0);
    dgemm_tn(m, ncols, n, H.Y.data() + i0, n, in, ncols, out + static_cast<std::size_t>(i0) * ncols, ncols);
  }
}

}  // namespace uppe
