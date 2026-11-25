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

// The simulation keeps E(w,kr) at 0 in rows i >= hcut[q] (Propagator::hcut); the test field does the same.
void zero_beyond_hcut(Field& f, const Propagator& P, int nt) {
  const int nq = nt / 2;
  for (int i = 0; i < P.nr; ++i)
    for (int q = 0; q < nq; ++q) {
      if (i < P.hcut[q]) continue;
      f.half[static_cast<std::size_t>(i) * nq + q] = cplx(0.0, 0.0);
      f.full[static_cast<std::size_t>(nq + q) * P.nr + i] = cplx(0.0, 0.0);
      if (q > 0) f.full[static_cast<std::size_t>(nq - q) * P.nr + i] = cplx(0.0, 0.0);
    }
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
  // The truncation must be exercised wherever the grid reaches past 2*krmax of the lowest bin, beyond which
  // the filter is exactly 0. krmax in units of 1/a0 grows with a0, so at the f75 a0 only the THz test grid
  // does; the truncation code itself does not depend on a0.
  if (g.kr.back() > 2.0 * b.krmax[nq + 1] && g.kr.front() < b.krmax[nq + 1]) CHECK(partial > 0);

  // qmax: the last bin with wfilter != 0. hcut: rows where kplasma*F*wfilter can be nonzero, and no more.
  CHECK(P.qmax >= 0 && P.qmax <= nq);
  CHECK(static_cast<int>(P.hcut.size()) == nq);
  bool qmaxOk = true, hcutZero = true, hcutTight = true;
  for (int q = 0; q < nq; ++q) {
    qmaxOk = qmaxOk && ((q < P.qmax) == (P.wfilter[q] != 0.0));
    const int hc = P.hcut[q];
    hcutZero = hcutZero && hc >= 0 && hc <= P.cut[q] && (q < P.qmax || hc == 0);
    for (int i = hc; i < P.nr; ++i) {
      const double kp = P.kplasma[static_cast<std::size_t>(i) * nq + q];
      hcutZero = hcutZero && kp * kr_filter(g.kr[i], b.krmax[nq + q]) * P.wfilter[q] == 0.0;
    }
    if (hc > 0) {
      const double kp = P.kplasma[static_cast<std::size_t>(hc - 1) * nq + q];
      hcutTight = hcutTight && kp != 0.0 && kr_filter(g.kr[hc - 1], b.krmax[nq + q]) != 0.0;
    }
  }
  CHECK(qmaxOk);
  CHECK(hcutZero);
  CHECK(hcutTight);
}

// hcut below cut somewhere: the evanescent rows (kz^2 <= 0) are cut off before the filter.
bool hcut_below_cut(const Propagator& P) {
  for (int q = 0; q < P.nq; ++q)
    if (P.hcut[q] < P.cut[q]) return true;
  return false;
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
    CHECK(hcut_below_cut(P));

    Field E = random_field(rng, g.size(), tf.ntT);
    zero_beyond_hcut(E, P, tf.ntT);
    linear_step(E.half.data(), P, g, b, 0, dz);
    ref::lin_thread(E.full, K, g, b, 0, dz);
    CHECK(compare_half(E.half, E.full, g.size(), tf.ntT) < 1e-11);

    // kdivkz needs no such field: the rows it skips are multiplied by exactly 0.
    Field k = random_field(rng, g.size(), tf.ntT);
    kdivkz(k.half.data(), P, g, b);
    ref::kdivkz(k.full, K, g, b, c::omega_max_THz);
    CHECK(compare_half(k.half, k.full, g.size(), tf.ntT) < 1e-11);

    // A low omega_max puts the upper bins beyond qmax, as 125 THz does on the production THz grid.
    const double omega_low = b.freqs[b.nq() + b.nq() / 2];
    const Propagator P2 = make_propagator(b, g, vg, omega_low);
    check_propagator(P2, b, g, K, omega_low);
    CHECK(P2.qmax > 1 && P2.qmax < b.nq());

    Field E2 = random_field(rng, g.size(), tf.ntT);
    zero_beyond_hcut(E2, P2, tf.ntT);
    linear_step(E2.half.data(), P2, g, b, 0, dz);
    ref::lin_thread(E2.full, K, g, b, 0, dz);
    CHECK(compare_half(E2.half, E2.full, g.size(), tf.ntT) < 1e-11);

    Field k2 = random_field(rng, g.size(), tf.ntT);
    kdivkz(k2.half.data(), P2, g, b);
    ref::kdivkz(k2.full, K, g, b, omega_low);
    CHECK(compare_half(k2.half, k2.full, g.size(), tf.ntT) < 1e-11);
  }

  // Optical band, propagated from bin qStart only.
  {
    // nR = nR_opt = 20 puts kr up to j_19/Rmax (about 9.8 at Rmax = 6, below krmax of every optical bin).
    const RadialGrid g = make_radial(roots, 20, 20, RadialKind::Optical);
    const Band b = make_band(tf.ntO, tf.dw, tf.wmin);
    const Propagator P = make_propagator(b, g, vg, c::omega_max);
    ref::Kz K = ref::kz_tables(b, g, vg);
    check_propagator(P, b, g, K, c::omega_max);

    const int qStart = 5;
    Field E = random_field(rng, g.size(), tf.ntO);
    zero_beyond_hcut(E, P, tf.ntO);
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
