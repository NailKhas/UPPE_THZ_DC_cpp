#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "check.hpp"
#include "first_step.hpp"
#include "grids.hpp"
#include "hankel.hpp"
#include "reference.hpp"

using namespace uppe;
namespace fs = std::filesystem;

int main() {
  test::init();

  // ntO = 2048 (nq = 1024) keeps the carrier bin int(w0*t0/dw) = 543 inside the band, as in production.
  const TimeFreq tf = make_time_freq(131072);
  const int nq = tf.nqO;
  const int it_inc = static_cast<int>(c::w0 * c::t0 / tf.dw);
  CHECK(it_inc > 0 && it_inc < nq);

  const int nR = 50, nR_opt = 10;
  const int n = nR - 1;
  const int nrKeep = nR_opt - 1;
  const std::vector<double> roots = bessel_j0_roots(nR + 5);
  const RadialGrid g = make_radial(roots, nR, nR_opt, RadialKind::First);
  const HankelMatrix H = make_hankel(roots, nR);
  const ref::Hankel RH = ref::hankel_matrix(roots, nR);
  const Band b = make_band(tf.ntO, tf.dw, tf.wmin);
  const double vg = 1.0003;
  const double dz = c::zsic / c::Ldifr;
  const double dz0 = 400. / c::Ldifr;

  const fs::path out = fs::temp_directory_path() / "uppe_test_first_step";
  fs::remove_all(out);
  fs::create_directories(out);

  std::vector<cplx> eOut(static_cast<std::size_t>(nrKeep) * nq, cplx(0.0, 0.0));
  first_step(tf, b, g, H, vg, dz, dz0, nrKeep, eOut.data(), out);

  ref::Kz K = ref::kz_tables(b, g, vg);
  const ref::Vec R = ref::first_step(tf, b, g, RH, K, dz, dz0);

  std::vector<cplx> expect(eOut.size());
  int active = 0;
  for (int q = 0; q < nq; ++q) {
    bool nonzero = false;
    for (int i = 0; i < nrKeep; ++i) {
      const cplx v = R[static_cast<std::size_t>(nq + q) * n + i];
      expect[static_cast<std::size_t>(i) * nq + q] = v;
      nonzero = nonzero || v != cplx(0.0, 0.0);
    }
    if (nonzero) ++active;
  }
  CHECK(active > 10);
  CHECK(test::rel_err(eOut.data(), expect.data(), eOut.size()) < 1e-10);

  // SpSIC.dat: one line per active bin. The 1e-30 intensity threshold sits at the FFT round-off floor
  // (|O| ~ 1e-15 |O|max), so bins within two decades of it may be selected differently by MKL/FFTW and
  // the reference DFT; only bins clearly above or below the threshold are checked.
  {
    ref::Vec O(tf.ntO);
    for (int it = 0; it < tf.ntO; ++it) {
      const double t = it * tf.dt + tf.tmin;
      O[it] = std::exp(-.5 * t * t) * std::cos(c::w0 * c::t0 * t);
    }
    O = ref::fortran_forward(O);
    double ISm = 0.0;
    for (int q = 0; q < nq; ++q) ISm = std::max(ISm, std::norm(O[nq + q]));

    std::ifstream f(out / "SpSIC.dat");
    CHECK(static_cast<bool>(f));
    std::vector<bool> listed(nq, false);
    long long idx = 0;
    double fr = 0.0, I = 0.0;
    while (f >> idx >> fr >> I) {
      const long long q = idx - nq;
      CHECK(q >= 1 && q < nq);
      if (q >= 1 && q < nq) {
        CHECK(!listed[q]);
        listed[q] = true;
        CHECK(std::norm(O[nq + q]) > ISm * 1e-32);
      }
    }
    for (int q = 1; q < nq; ++q)
      if (std::norm(O[nq + q]) > ISm * 1e-28) CHECK(listed[q]);
  }

  // Incident.dat: r*a0 and |Ebuf(it,r)|^2 for all n radial points.
  {
    std::ifstream f(out / "Incident.dat");
    CHECK(static_cast<bool>(f));
    std::vector<double> got, want;
    double r = 0.0, I = 0.0;
    while (f >> r >> I) got.push_back(I);
    for (int i = 0; i < n; ++i) want.push_back(std::norm(R[static_cast<std::size_t>(nq + it_inc) * n + i]));
    CHECK(static_cast<int>(got.size()) == n);
    if (static_cast<int>(got.size()) == n) CHECK(test::rel_err(got.data(), want.data(), got.size()) < 1e-9);
  }

  // dz0 = 0: the lens acts on the source profile. The reference still applies the angular filter and two
  // Hankel transforms, which change the field only at the round-off level of the transform pair.
  {
    std::vector<cplx> e0(eOut.size(), cplx(0.0, 0.0));
    first_step(tf, b, g, H, vg, dz, 0.0, nrKeep, e0.data(), out);
    ref::Kz K0 = ref::kz_tables(b, g, vg);
    const ref::Vec R0 = ref::first_step(tf, b, g, RH, K0, dz, 0.0);
    std::vector<cplx> expect0(e0.size());
    for (int q = 0; q < nq; ++q)
      for (int i = 0; i < nrKeep; ++i)
        expect0[static_cast<std::size_t>(i) * nq + q] = R0[static_cast<std::size_t>(nq + q) * n + i];
    CHECK(test::rel_err(e0.data(), expect0.data(), e0.size()) < 1e-8);
  }

  fs::remove_all(out);
  return test::finish("test_first_step");
}
