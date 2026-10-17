#include "propagation.hpp"

namespace uppe {

void linear_step(cplx* E, const Propagator& P, const RadialGrid& g, const Band& b, int qStart, double dz) {
  const int nr = P.nr;
  const int nq = P.nq;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nr; ++i) {
    cplx* row = E + static_cast<std::size_t>(i) * nq;
    const double* kap = P.kappa.data() + static_cast<std::size_t>(i) * nq;
    const double kr = g.kr[i];
    for (int q = qStart; q < nq; ++q) {
      if (i >= P.cut[q]) {
        row[q] = cplx(0.0, 0.0);
        continue;
      }
      const double ph = -kap[q] * dz;
      const double F = kr_filter(kr, b.krmax[nq + q]);
      row[q] = cmul(row[q], cplx(std::cos(ph), std::sin(ph))) * F;
    }
  }
}

void kdivkz(cplx* K, const Propagator& P, const RadialGrid& g, const Band& b) {
  const int nr = P.nr;
  const int nq = P.nq;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nr; ++i) {
    cplx* row = K + static_cast<std::size_t>(i) * nq;
    const double* kp = P.kplasma.data() + static_cast<std::size_t>(i) * nq;
    const double kr = g.kr[i];
    for (int q = 0; q < nq; ++q) {
      if (i >= P.cut[q]) {
        row[q] = cplx(0.0, 0.0);
        continue;
      }
      const double F = kr_filter(kr, b.krmax[nq + q]);
      row[q] = row[q] * kp[q] * F * P.wfilter[q];
    }
  }
}

}  // namespace uppe
