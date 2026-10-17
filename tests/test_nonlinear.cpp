#include <vector>

#include "check.hpp"
#include "nonlinear.hpp"
#include "reference.hpp"

using namespace uppe;

namespace {

struct Case {
  double etAmp;
  bool expectClamp;
};

// kbuf is built from spectra of real signals times omega^2*j, omega and j, so the negative
// frequencies satisfy S(n/2 - q) = -conj(S(n/2 + q)); KdivKZ discards them (k(nt-iit) = conj(k(iit))),
// which is why only the non-negative half is computed.
double antihermitian_defect(const cplx* row, int n, int qFrom) {
  double dmax = 0.0, smax = 0.0;
  for (int q = qFrom; q < n / 2; ++q) {
    dmax = std::max(dmax, std::abs(row[n / 2 - q] + std::conj(row[n / 2 + q])));
    smax = std::max(smax, std::abs(row[n / 2 + q]));
  }
  return smax > 0.0 ? dmax / smax : dmax;
}

void run_case(const Case& cs, test::Rng& rng) {
  const TimeFreq tf = make_time_freq(4096);
  const int nr = 3;
  const double dz = 0.013;

  Physics ph = make_physics();
  ph.P_N2.resize(c::ion_table_size);
  ph.P_O2.resize(c::ion_table_size);
  for (int i = 0; i < c::ion_table_size; ++i) {
    ph.P_N2[i] = 1e-4 * (1.0 + i) / (1.0 + 1e-3 * i);
    ph.P_O2[i] = 3e-4 * std::sqrt(1.0 + i);
  }

  // THz field (w,r) as the stored half, and its full Fortran-order version.
  std::vector<cplx> thz(static_cast<std::size_t>(nr) * tf.nqT, cplx(0.0, 0.0));
  ref::Vec thzFull;
  thzFull.reserve(static_cast<std::size_t>(nr) * tf.ntT);
  for (int ir = 0; ir < nr; ++ir) {
    cplx* row = thz.data() + static_cast<std::size_t>(ir) * tf.nqT;
    for (int q = 1; q < tf.nqT; ++q) row[q] = rng.cuniform(0.05);
    const ref::Vec f = ref::full_from_half(row, tf.ntT);
    thzFull.insert(thzFull.end(), f.begin(), f.end());
  }

  // Optical field (t,r): a chirped pulse reaching above the ionization threshold.
  std::vector<double> et(static_cast<std::size_t>(nr) * tf.ntO);
  ref::Vec etFull(et.size());
  for (int ir = 0; ir < nr; ++ir)
    for (int t = 0; t < tf.ntO; ++t) {
      const double x = (t - tf.ntO / 2) / 9.0;
      const double v = cs.etAmp * (1.0 - 0.2 * ir) * std::exp(-x * x) * std::cos(1.3 * t + 0.02 * t * t) +
                       rng.uniform(-0.01, 0.01);
      et[static_cast<std::size_t>(ir) * tf.ntO + t] = v;
      etFull[static_cast<std::size_t>(ir) * tf.ntO + t] = v;
    }

  NonlinearWorkspace ws(tf, max_threads());
  std::vector<cplx> kOpt(static_cast<std::size_t>(nr) * tf.nqO, cplx(9.0, 9.0));
  std::vector<cplx> kThz(static_cast<std::size_t>(nr) * tf.nqT, cplx(9.0, 9.0));
  std::vector<double> neMax(nr, 0.0);
  NonlinearInputs in;
  in.tf = &tf;
  in.ph = &ph;
  in.thz_r = thz.data();
  in.et = et.data();
  in.nr = nr;
  in.dz = dz;
  in.k_opt = kOpt.data();
  in.k_thz = kThz.data();
  in.ne_max = neMax.data();
  const long long clamped = nonlinear_response(in, ws);

  const ref::JefResult R = ref::jef_thread(tf, ph, thzFull, etFull, nr, dz);

  CHECK(R.ionizing_samples > 0);
  CHECK(clamped == R.clamped);
  CHECK((clamped > 0) == cs.expectClamp);

  std::vector<cplx> expThz(kThz.size()), expOpt(kOpt.size());
  for (int ir = 0; ir < nr; ++ir) {
    const cplx* rt = R.k_thz.data() + static_cast<std::size_t>(ir) * tf.ntT;
    const cplx* ro = R.k.data() + static_cast<std::size_t>(ir) * tf.ntO;
    CHECK(antihermitian_defect(rt, tf.ntT, 1) < 1e-9);
    // The zeroed THz window kbuf(nt_all/2-ntT/2 : nt_all/2+ntT/2-1) is asymmetric: it removes the
    // optical bin p = -nqT/ratio but keeps p = +nqT/ratio.
    const int pEdge = tf.nqT / tf.ratio;
    CHECK(antihermitian_defect(ro, tf.ntO, pEdge + 1) < 1e-9);
    CHECK(ro[tf.nqO - pEdge] == cplx(0.0, 0.0));
    CHECK(ro[tf.nqO + pEdge] != cplx(0.0, 0.0));
    CHECK(ro[0] == cplx(0.0, 0.0));  // kbuf(0)
    for (int q = 0; q < tf.nqT; ++q) expThz[static_cast<std::size_t>(ir) * tf.nqT + q] = rt[tf.nqT + q];
    for (int p = 0; p < tf.nqO; ++p) expOpt[static_cast<std::size_t>(ir) * tf.nqO + p] = ro[tf.nqO + p];
    CHECK(neMax[ir] > 0.0);
    CHECK_CLOSE(neMax[ir], R.ne_max[ir], 1e-12 * R.ne_max[ir]);
  }
  CHECK(test::rel_err(kThz.data(), expThz.data(), kThz.size()) < 1e-9);
  CHECK(test::rel_err(kOpt.data(), expOpt.data(), kOpt.size()) < 1e-9);

  // ne_max is a running maximum: a second call with a larger stored value keeps it.
  std::vector<double> big(nr, 10.0);
  in.ne_max = big.data();
  nonlinear_response(in, ws);
  for (int ir = 0; ir < nr; ++ir) CHECK(big[ir] == 10.0);
}

}  // namespace

int main() {
  test::init();
  test::Rng rng;
  run_case({3.0, false}, rng);
  run_case({400.0, true}, rng);
  return test::finish("test_nonlinear");
}
