#include "grids.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "bessel.hpp"
#include "output.hpp"

namespace uppe {

std::vector<double> bessel_j0_roots(int count) {
  if (count < 1) throw std::invalid_argument("bessel_j0_roots: count must be positive");
  const double true_pi = std::acos(-1.0);
  std::vector<double> roots(count);
  std::vector<double> guesses(count);

#pragma omp parallel for schedule(static)
  for (int n = 1; n <= count; ++n) {
    // McMahon expansion as the starting point, then the Newton iteration of PrepareHankel.
    const double beta = (n - 0.25) * true_pi;
    const double b = 8.0 * beta;
    const double guess = beta + 1.0 / b - 124.0 / (3.0 * b * b * b) + 120928.0 / (15.0 * b * b * b * b * b);
    guesses[n - 1] = guess;
    double r_1 = guess;
    double r_2 = r_1 - bessel_j0(r_1) / (-bessel_j1(r_1));
    // Above ~8000 one ulp exceeds err, so convergence means r_2 == r_1; the cap guards a 1-ulp cycle.
    for (int it = 0; it < 100 && std::fabs(r_2 - r_1) > c::err; ++it) {
      r_1 = r_2;
      r_2 = r_1 - bessel_j0(r_1) / (-bessel_j1(r_1));
    }
    roots[n - 1] = (r_1 + r_2) / 2.0;
  }

  for (int n = 1; n <= count; ++n) {
    const double r = roots[n - 1];
    const double prev = n > 1 ? roots[n - 2] : 0.0;
    const bool ok = std::isfinite(r) && std::fabs(r - guesses[n - 1]) < 0.1 && r - prev > 2.0 &&
                    bessel_j0(r - 0.01) * bessel_j0(r + 0.01) < 0.0;
    if (!ok) throw std::runtime_error("Bessel root search failed at n = " + std::to_string(n));
  }
  return roots;
}

RadialGrid make_radial(const std::vector<double>& roots, int nR, int nR_opt, RadialKind kind) {
  if (nR < 3 || nR_opt < 1 || static_cast<int>(roots.size()) < nR || static_cast<int>(roots.size()) < nR_opt)
    throw std::invalid_argument("make_radial: not enough Bessel roots");
  const int n = nR - 1;
  const double jN = roots[nR - 1];
  const double jO = roots[nR_opt - 1];
  RadialGrid g;
  g.r1.resize(n);
  g.dr1.resize(n);
  g.kr.resize(n);
  for (int i = 0; i < n; ++i) {
    const double ji = roots[i];
    switch (kind) {
      case RadialKind::First:
        g.r1[i] = c::Rmax * ji / jO;
        g.kr[i] = ji / c::Rmax / (jN / jO);
        break;
      case RadialKind::Optical:
        g.r1[i] = c::Rmax / jO * ji;
        g.kr[i] = jO / c::Rmax / jN * ji;
        break;
      case RadialKind::THz:
        g.r1[i] = ji * c::Rmax / jO;
        g.kr[i] = ji / c::Rmax / jN * jO;
        break;
    }
  }
  for (int i = 0; i < n - 1; ++i) g.dr1[i] = g.r1[i + 1] - g.r1[i];
  g.dr1[n - 1] = g.dr1[n - 2];
  return g;
}

Band make_band(int nt, double dw, double wmin) {
  if (nt < 4 || nt % 2 != 0) throw std::invalid_argument("make_band: nt must be even");
  Band b;
  b.nt = nt;
  b.freqs.assign(nt, 0.0);
  b.refrs.assign(nt, 0.0);
  b.krmax.assign(nt, 0.0);
  b.fr.assign(nt, 0.0f);
  const double sin_alpha = std::sin(c::alpha_max);
  for (int iit = 1; iit < nt; ++iit) {
    const double f = dw * iit + wmin;
    b.freqs[iit] = f;
    b.refrs[iit] = 1.993e-4 + 5.58e-37 / (c::t0 * c::t0) * (f * f);
    b.krmax[iit] = f * c::a0 / c::cL / c::t0 * sin_alpha;
  }
  for (int iit = 0; iit < nt; ++iit) b.fr[iit] = static_cast<float>(b.freqs[iit] / c::t0 * 1e-12 / 2. / c::pi);
  return b;
}

Dispersion compute_dispersion(const Band& b, const TimeFreq& tf) {
  const int it = static_cast<int>((c::w0 * c::t0 - tf.wmin) / tf.dw);
  if (it - 16 < 1 || it + 16 >= b.nt)
    throw std::runtime_error("compute_dispersion: the carrier frequency is outside the optical band");
  Dispersion d;
  d.ng = (b.refrs[it + 16] - b.refrs[it - 16]) / (32 * tf.dw) * b.freqs[it] + b.refrs[it];
  d.d2kdw2 = (b.refrs[it + 16] * b.freqs[it + 16] + b.refrs[it - 16] * b.freqs[it - 16] -
              2 * b.refrs[it] * b.freqs[it]) /
             sq(16 * tf.dw);
  d.vg = 1.0 + d.ng;
  return d;
}

Propagator make_propagator(const Band& band, const RadialGrid& grid, double vg, double omega_max) {
  Propagator p;
  p.nr = grid.size();
  p.nq = band.nq();
  const int nr = p.nr;
  const int nq = p.nq;
  p.kappa.resize(static_cast<std::size_t>(nr) * nq);
  p.kplasma.resize(static_cast<std::size_t>(nr) * nq);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < nr; ++i) {
    double* ka = p.kappa.data() + static_cast<std::size_t>(i) * nq;
    double* kp = p.kplasma.data() + static_cast<std::size_t>(i) * nq;
    for (int q = 1; q < nq; ++q) {
      const int iit = nq + q;
      const KzValue v = kz_value(band.freqs[iit], band.refrs[iit], grid.kr[i], vg);
      ka[q] = v.kappa;
      kp[q] = v.kplasma;
    }
  }

  p.cut.assign(nq, 0);
  p.wfilter.assign(nq, 0.0);
#pragma omp parallel for schedule(dynamic, 64)
  for (int q = 0; q < nq; ++q) {
    const double krmax = band.krmax[nq + q];
    int cut = 0;
    for (int i = nr - 1; i >= 0; --i) {
      if (kr_filter(grid.kr[i], krmax) != 0.0) {
        cut = i + 1;
        break;
      }
    }
    p.cut[q] = cut;
    p.wfilter[q] = std::exp(-pow20(band.freqs[nq + q] / omega_max));
  }
  return p;
}

void write_radial_files(const std::filesystem::path& outDir, const RadialGrid& g, const char* nameUm,
                        const char* name, const char* nameKr) {
  const std::filesystem::path res = outDir / "res";
  TextWriter fum(res / nameUm);
  TextWriter fa0(res / name);
  TextWriter fkr(res / nameKr);
  fum.text("r(um) dr(um)").nl();
  fa0.text("r(a0) dr(a0)").nl();
  fkr.text("kr(1/a0)").nl();
  for (int i = 0; i < g.size(); ++i) {
    fum.d(g.r1[i] * c::a0 * 1e4).d(g.dr1[i] * c::a0 * 1e4).nl();
    fa0.d(g.r1[i]).d(g.dr1[i]).nl();
    fkr.d(g.kr[i]).nl();
  }
}

void write_band_files(const std::filesystem::path& outDir, const Band& band, const Propagator& prop, BandKind kind) {
  const std::filesystem::path res = outDir / "res";
  const bool opt = kind == BandKind::Optical;
  TextWriter fw(res / (opt ? "wO2.dat" : "wT2.dat"));
  TextWriter fr(res / (opt ? "wrO2.dat" : "wrT2.dat"));
  TextWriter fk(res / (opt ? "krOmax2.dat" : "krTmax2.dat"));
  TextWriter fz(res / (opt ? "w_kzO2.dat" : "w_kzT2.dat"));
  fw.text(opt ? "it Frequency(t0^-1) Frequency(THz)" : "it Frequency(t0^-1)").nl();
  fr.text("Frequency(THz) Refraction").nl();
  fk.text("Frequency(THz) krmax(1/a0)").nl();
  fz.text("Frequency(THz) kz(w,kr->0)").nl();
  const int nt = band.nt;
  const int nq = band.nq();
  for (int iit = 1; iit < nt; ++iit) {
    fw.i(iit).d(band.freqs[iit]);
    if (opt) fw.f(band.fr[iit]);
    fw.nl();
    fr.f(band.fr[iit]).d(band.refrs[iit]).nl();
    fk.f(band.fr[iit]).d(band.krmax[iit]).nl();
    // kz(iit,1) on the full axis: kz(nt-iit) = -kz(iit), kz(nt/2) = 0.
    double kzi = 0.0;
    if (iit > nq) {
      kzi = prop.kappa[iit - nq];
    } else if (iit < nq) {
      kzi = -prop.kappa[nt - iit - nq];
    }
    fz.f(band.fr[iit]).d(0.0).d(kzi).nl();
  }
}

void write_collision_file(const std::filesystem::path& outDir, const TimeFreq& tf) {
  TextWriter f(outDir / "res" / "wA_coll2.dat");
  f.text("Frequency Re[1/(iw+nu)] Im[1/(iw+nu)]").nl();
  for (int m = 1; m < tf.nt_all; ++m) {
    const double omega = tf.dw_THz * m + tf.wmin;
    const cplx inv = cplx(1.0, 0.0) / cplx(c::omega_collide, omega);
    f.d(omega).d(inv.real()).d(inv.imag()).nl();
  }
}

}  // namespace uppe
