#include <vector>

#include "check.hpp"
#include "grids.hpp"
#include "propagation.hpp"
#include "reference.hpp"

using namespace uppe;

namespace {

// Random [nr][nq] half spectrum and the matching full Fortran-order array [nt][nr].
struct Field {
  std::vector<cplx> half;
  ref::Vec full;
};

Field random_field(test::Rng& rng, int nr, int nt) {
  const int nq = nt / 2;
  Field f;
  f.half.resize(static_cast<std::size_t>(nr) * nq);
  f.full.assign(static_cast<std::size_t>(nt) * nr, cplx(0.0, 0.0));
  for (int i = 0; i < nr; ++i)
    for (int q = 0; q < nq; ++q) {
      const cplx v = q == 0 ? cplx(0.0, 0.0) : rng.cuniform(1.0);
      f.half[static_cast<std::size_t>(i) * nq + q] = v;
      f.full[static_cast<std::size_t>(nq + q) * nr + i] = v;
      if (q > 0) f.full[static_cast<std::size_t>(nq - q) * nr + i] = std::conj(v);
    }
  return f;
}

double compare_half(const std::vector<cplx>& half, const ref::Vec& full, int nr, int nt) {
  const int nq = nt / 2;
  std::vector<cplx> expect(half.size());
  for (int i = 0; i < nr; ++i)
    for (int q = 0; q < nq; ++q)
      expect[static_cast<std::size_t>(i) * nq + q] = full[static_cast<std::size_t>(nq + q) * nr + i];
  return test::rel_err(half.data(), expect.data(), half.size());
}

void check_propagator(const Propagator& P, const Band& b, const RadialGrid& g, ref::Kz& K, double omega_max) {
  const int nq = P.nq;
  double kapErr = 0.0, kapMax = 0.0, kpErr = 0.0, kpMax = 0.0, reMax = 0.0;
  for (int i = 0; i < P.nr; ++i)
    for (int q = 0; q < nq; ++q) {
      const cplx kz = K.at(K.kz, nq + q, i);
      const cplx kp = K.at(K.kp, nq + q, i);
      reMax = std::max(reMax, std::fabs(kz.real()));
      kapErr = std::max(kapErr, std::fabs(P.kappa[static_cast<std::size_t>(i) * nq + q] - kz.imag()));
      kapMax = std::max(kapMax, std::fabs(kz.imag()));
      kpErr = std::max(kpErr, std::fabs(P.kplasma[static_cast<std::size_t>(i) * nq + q] - kp.real()));
      kpMax = std::max(kpMax, std::fabs(kp.real()));
    }
  CHECK(reMax == 0.0);
  CHECK(kapErr <= 1e-14 * kapMax);
  CHECK(kpErr <= 1e-14 * kpMax);

  bool cutOk = true;
  int partial = 0;
  for (int q = 0; q < nq; ++q) {
    const int cut = P.cut[q];
    for (int i = 0; i < P.nr; ++i) {
      const double F = std::exp(-std::pow(g.kr[i] / b.krmax[nq + q], 20));
      if (i >= cut) cutOk = cutOk && F == 0.0;
    }
    if (cut > 0) cutOk = cutOk && kr_filter(g.kr[cut - 1], b.krmax[nq + q]) != 0.0;
    if (cut > 0 && cut < P.nr) ++partial;
    CHECK_CLOSE(P.wfilter[q], std::exp(-std::pow(b.freqs[nq + q] / omega_max, 20)), 1e-14);
  }
  CHECK(cutOk);
  CHECK(P.cut[0] == 0);
  // The test grid must exercise the truncation. With a longer focus a0, and krmax in units of 1/a0,
  // grow and the small grid stays below the cut; the truncation code does not depend on the focus.
  if (c::focus == 20.0) CHECK(partial > 0);
}

}  // namespace

int main() {
  test::init();
  test::Rng rng;

  const TimeFreq tf = make_time_freq(4096);
  CHECK(tf.ntO == 64 && tf.ntT == 128);
  const std::vector<double> roots = bessel_j0_roots(40);
  const double vg = 1.0003;
  const double dz = 0.037;

  // THz band.
  {
    const RadialGrid g = make_radial(roots, 30, 10, RadialKind::THz);
    const Band b = make_band(tf.ntT, tf.dw_THz, tf.wmin_THz);
    const Propagator P = make_propagator(b, g, vg, c::omega_max_THz);
    ref::Kz K = ref::kz_tables(b, g, vg);
    check_propagator(P, b, g, K, c::omega_max_THz);

    Field E = random_field(rng, g.size(), tf.ntT);
    linear_step(E.half.data(), P, g, b, 0, dz);
    ref::lin_thread(E.full, K, g, b, 0, dz);
    CHECK(compare_half(E.half, E.full, g.size(), tf.ntT) < 1e-11);

    Field k = random_field(rng, g.size(), tf.ntT);
    kdivkz(k.half.data(), P, g, b);
    ref::kdivkz(k.full, K, g, b, c::omega_max_THz);
    CHECK(compare_half(k.half, k.full, g.size(), tf.ntT) < 1e-11);
  }

  // Optical band, propagated from bin qStart only.
  {
    // nR = nR_opt = 20 puts kr up to ~2.9, beyond krmax of the lowest optical bins.
    const RadialGrid g = make_radial(roots, 20, 20, RadialKind::Optical);
    const Band b = make_band(tf.ntO, tf.dw, tf.wmin);
    const Propagator P = make_propagator(b, g, vg, c::omega_max);
    ref::Kz K = ref::kz_tables(b, g, vg);
    check_propagator(P, b, g, K, c::omega_max);

    const int qStart = 5;
    Field E = random_field(rng, g.size(), tf.ntO);
    const std::vector<cplx> before = E.half;
    linear_step(E.half.data(), P, g, b, qStart, dz);
    ref::lin_thread(E.full, K, g, b, qStart, dz);
    CHECK(compare_half(E.half, E.full, g.size(), tf.ntO) < 1e-11);
    bool untouched = true;
    for (int i = 0; i < g.size(); ++i)
      for (int q = 0; q < qStart; ++q) {
        const std::size_t idx = static_cast<std::size_t>(i) * tf.nqO + q;
        untouched = untouched && E.half[idx] == before[idx];
      }
    CHECK(untouched);

    Field k = random_field(rng, g.size(), tf.ntO);
    kdivkz(k.half.data(), P, g, b);
    ref::kdivkz(k.full, K, g, b, c::omega_max);
    CHECK(compare_half(k.half, k.full, g.size(), tf.ntO) < 1e-11);
  }

  return test::finish("test_propagation");
}
