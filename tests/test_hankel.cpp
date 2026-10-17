#include <vector>

#include "check.hpp"
#include "grids.hpp"
#include "hankel.hpp"
#include "reference.hpp"

using namespace uppe;

namespace {

const int nR = 40;
const int n = nR - 1;
const int nq = 37;  // not a multiple of the block width

std::vector<cplx> random_field(test::Rng& rng, int rows) {
  std::vector<cplx> a(static_cast<std::size_t>(n) * nq, cplx(0.0, 0.0));
  for (int k = 0; k < rows; ++k)
    for (int q = 0; q < nq; ++q) a[static_cast<std::size_t>(k) * nq + q] = rng.cuniform(1.0);
  return a;
}

ref::Vec column(const std::vector<cplx>& a, int q) {
  ref::Vec v(n);
  for (int k = 0; k < n; ++k) v[k] = a[static_cast<std::size_t>(k) * nq + q];
  return v;
}

// max over (i, q) with i < rowLimit(q) of |out - ref| / max |ref|
template <class RowLimit>
double compare(const std::vector<cplx>& out, const std::vector<ref::Vec>& R, RowLimit rowLimit) {
  double dmax = 0.0, rmax = 0.0;
  for (int q = 0; q < nq; ++q)
    for (int i = 0; i < rowLimit(q); ++i) {
      dmax = std::max(dmax, std::abs(out[static_cast<std::size_t>(i) * nq + q] - R[q][i]));
      rmax = std::max(rmax, std::abs(R[q][i]));
    }
  return rmax > 0.0 ? dmax / rmax : dmax;
}

}  // namespace

int main() {
  test::init();
  test::Rng rng;

  const std::vector<double> roots = bessel_j0_roots(nR + 5);
  const HankelMatrix H = make_hankel(roots, nR);
  const ref::Hankel R = ref::hankel_matrix(roots, nR);
  CHECK(H.n == n);

  const std::vector<cplx> a = random_field(rng, n);
  const std::vector<cplx> b = random_field(rng, n);
  auto all = [](int) { return n; };

  // Plain transform.
  {
    std::vector<cplx> out(a.size());
    hankel_spectral(H, HankelInput{a.data(), nullptr, 0.0, n}, out.data(), n, nq, nullptr, nullptr);
    std::vector<ref::Vec> Rq(nq);
    for (int q = 0; q < nq; ++q) Rq[q] = ref::hnk(R, column(a, q));
    CHECK(compare(out, Rq, all) < 1e-13);
  }

  // The same matrix maps r -> kr and kr -> r: for a Gaussian that decays well inside both grids,
  // two transforms return the input.
  {
    const int m = 1;
    std::vector<cplx> g(static_cast<std::size_t>(n) * m), Gk(g.size()), back(g.size());
    for (int k = 0; k < n; ++k) {
      const double x = 5.0 * roots[k] / roots[nR - 1];
      g[k] = cplx(std::exp(-x * x), 0.5 * std::exp(-x * x));
    }
    hankel_spectral(H, HankelInput{g.data(), nullptr, 0.0, n}, Gk.data(), n, m, nullptr, nullptr);
    hankel_spectral(H, HankelInput{Gk.data(), nullptr, 0.0, n}, back.data(), n, m, nullptr, nullptr);
    CHECK(test::rel_err(back.data(), g.data(), g.size()) < 1e-6);
  }

  // Fused input a + beta*b.
  {
    const double beta = 0.5;
    std::vector<cplx> out(a.size());
    hankel_spectral(H, HankelInput{a.data(), b.data(), beta, n}, out.data(), n, nq, nullptr, nullptr);
    std::vector<ref::Vec> Rq(nq);
    for (int q = 0; q < nq; ++q) {
      ref::Vec v = column(a, q);
      const ref::Vec w = column(b, q);
      for (int k = 0; k < n; ++k) v[k] += beta * w[k];
      Rq[q] = ref::hnk(R, v);
    }
    CHECK(compare(out, Rq, all) < 1e-13);
  }

  // sumCut: input rows k >= cut[q] are zero and may be skipped.
  {
    std::vector<int> cut(nq);
    for (int q = 0; q < nq; ++q) cut[q] = static_cast<int>(rng.uniform(0.0, n + 0.999));
    cut[0] = 0;
    cut[nq - 1] = n;
    std::vector<cplx> az = a;
    for (int q = 0; q < nq; ++q)
      for (int k = cut[q]; k < n; ++k) az[static_cast<std::size_t>(k) * nq + q] = 0.0;
    std::vector<cplx> out(a.size(), cplx(777.0, 777.0));
    hankel_spectral(H, HankelInput{az.data(), nullptr, 0.0, n}, out.data(), n, nq, cut.data(), nullptr);
    std::vector<ref::Vec> Rq(nq);
    for (int q = 0; q < nq; ++q) Rq[q] = ref::hnk(R, column(az, q));
    CHECK(compare(out, Rq, all) < 1e-13);
  }

  // Fewer input rows than n, fewer output rows M, and outCut.
  {
    const int rows = 15, M = 20;
    std::vector<int> cut(nq);
    for (int q = 0; q < nq; ++q) cut[q] = static_cast<int>(rng.uniform(0.0, n + 0.999));
    cut[3] = 0;
    const std::vector<cplx> ar = random_field(rng, rows);
    const cplx sentinel(777.0, -777.0);
    std::vector<cplx> out(a.size(), sentinel);
    hankel_spectral(H, HankelInput{ar.data(), nullptr, 0.0, rows}, out.data(), M, nq, nullptr, cut.data());
    std::vector<ref::Vec> Rq(nq);
    for (int q = 0; q < nq; ++q) Rq[q] = ref::hnk(R, column(ar, q));
    CHECK(compare(out, Rq, [&](int q) { return std::min(M, cut[q]); }) < 1e-13);

    double rmax = 0.0;
    for (int q = 0; q < nq; ++q)
      for (int i = 0; i < n; ++i) rmax = std::max(rmax, std::abs(Rq[q][i]));
    bool untouched = true, zeroOrExact = true;
    for (int q = 0; q < nq; ++q) {
      for (int i = M; i < n; ++i) untouched = untouched && out[static_cast<std::size_t>(i) * nq + q] == sentinel;
      for (int i = std::min(M, cut[q]); i < M; ++i) {
        const cplx v = out[static_cast<std::size_t>(i) * nq + q];
        zeroOrExact = zeroOrExact && (v == cplx(0.0, 0.0) || std::abs(v - Rq[q][i]) <= 1e-13 * rmax);
      }
    }
    CHECK(untouched);
    CHECK(zeroOrExact);
  }

  // Dense real transform (FirstStep).
  {
    const int ncols = 6;
    std::vector<double> in(static_cast<std::size_t>(n) * ncols), out(in.size());
    for (double& v : in) v = rng.uniform(-1.0, 1.0);
    hankel_dense(H, in.data(), out.data(), ncols);
    std::vector<double> expect(in.size());
    for (int col = 0; col < ncols; ++col) {
      ref::Vec v(n);
      for (int k = 0; k < n; ++k) v[k] = in[static_cast<std::size_t>(k) * ncols + col];
      const ref::Vec s = ref::hnk(R, v);
      for (int i = 0; i < n; ++i) expect[static_cast<std::size_t>(i) * ncols + col] = s[i].real();
    }
    CHECK(test::rel_err(out.data(), expect.data(), out.size()) < 1e-13);
  }

  return test::finish("test_hankel");
}
