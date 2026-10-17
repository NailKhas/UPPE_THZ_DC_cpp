#include "first_step.hpp"

#include <stdexcept>
#include <vector>

#include "fft.hpp"
#include "output.hpp"

namespace uppe {

namespace {

// Field arrays are [n][nA]: radial point i, active frequency a.
void propagate(const cplx* in, cplx* out, int n, const std::vector<int>& active, const Band& b, const RadialGrid& g,
               double vg, double step) {
  const int nA = static_cast<int>(active.size());
  const int nq = b.nq();
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    for (int a = 0; a < nA; ++a) {
      const int iit = nq + active[a];
      const double kappa = kz_value(b.freqs[iit], b.refrs[iit], g.kr[i], vg).kappa;
      const double ph = -kappa * step;
      const double F = kr_filter(g.kr[i], b.krmax[iit]);
      const std::size_t idx = static_cast<std::size_t>(i) * nA + a;
      out[idx] = cmul(in[idx], cplx(std::cos(ph), std::sin(ph))) * F;
    }
  }
}

void lens(const cplx* in, cplx* out, int n, const std::vector<int>& active, const Band& b, const RadialGrid& g) {
  const int nA = static_cast<int>(active.size());
  const int nq = b.nq();
  const double foc_ph = .5 * (c::a0 * c::a0) / (c::focus * c::t0 * c::cL);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    const double r = g.r1[i];
    const double g8 = std::exp(-pow8(r * c::a0));
    for (int a = 0; a < nA; ++a) {
      const double th = foc_ph * (r * r) * b.freqs[nq + active[a]];
      const std::size_t idx = static_cast<std::size_t>(i) * nA + a;
      out[idx] = cmul(in[idx] * g8, cplx(std::cos(th), std::sin(th)));
    }
  }
}

}  // namespace

void first_step(const TimeFreq& tf, const Band& bandO, const RadialGrid& g, const HankelMatrix& H, double vg,
                double dz, double dz0, int nrKeep, cplx* eOut, const std::filesystem::path& outDir) {
  const int nt = tf.ntO;
  const int nq = nt / 2;
  const int n = H.n;
  if (g.size() != n || bandO.nt != nt || nrKeep < 1 || nrKeep > n)
    throw std::invalid_argument("first_step: inconsistent grids");

  // Incident pulse and its spectrum (non-negative frequencies).
  std::vector<cplx> O(nq);
  {
    RealFFT fft(nt);
    double* t = fft.time();
    for (int it = 0; it < nt; ++it) {
      const double tt = it * tf.dt + tf.tmin;
      t[it] = std::exp(-(.5 * (tt * tt))) * std::cos(c::w0 * c::t0 * tt);
    }
    centered_forward(fft, O.data(), nq);
  }

  double ISm = 0.0;
  for (int q = 0; q < nq; ++q) ISm = std::max(ISm, abs2(O[q]));

  std::vector<int> active;
  {
    TextWriter f(outDir / "SpSIC.dat");
    for (int q = 1; q < nq; ++q) {
      if (abs2(O[q]) > ISm * 1e-30) {
        active.push_back(q);
        f.i(nq + q).f(bandO.fr[nq + q]).d(abs2(O[q])).nl();
      }
    }
  }

  const int it_inc = static_cast<int>(c::w0 * c::t0 / tf.dw);  // Fortran it = int(w0*t0/dw)+nt/2
  const int nA = static_cast<int>(active.size());
  int a_inc = -1;
  for (int a = 0; a < nA; ++a)
    if (active[a] == it_inc) a_inc = a;

  AlignedArray<cplx> X(static_cast<std::size_t>(n) * std::max(nA, 1));
  AlignedArray<cplx> Y(static_cast<std::size_t>(n) * std::max(nA, 1));
  if (nA > 0) {
    const int ncols = 2 * nA;
    double* Xd = reinterpret_cast<double*>(X.data());
    double* Yd = reinterpret_cast<double*>(Y.data());
    const double a00_16 = pow16(c::a00);

    // SpOrigin = Origin(iit) * exp(-(r1*a0)**16/a00**16)
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      const double g16 = std::exp(-(pow16(g.r1[i] * c::a0) / a00_16));
      for (int a = 0; a < nA; ++a) X[static_cast<std::size_t>(i) * nA + a] = O[active[a]] * g16;
    }
    hankel_dense(H, Xd, Yd, ncols);
    propagate(Y.data(), X.data(), n, active, bandO, g, vg, dz0);
    hankel_dense(H, Xd, Yd, ncols);
    lens(Y.data(), X.data(), n, active, bandO, g);
    hankel_dense(H, Xd, Yd, ncols);
    propagate(Y.data(), X.data(), n, active, bandO, g, vg, dz);
    hankel_dense(H, Xd, Yd, ncols);

    for (int i = 0; i < nrKeep; ++i)
      for (int a = 0; a < nA; ++a)
        eOut[static_cast<std::size_t>(i) * nq + active[a]] = Y[static_cast<std::size_t>(i) * nA + a];
  }

  TextWriter f(outDir / "Incident.dat");
  for (int i = 0; i < n; ++i) {
    const double I = a_inc >= 0 ? abs2(Y[static_cast<std::size_t>(i) * nA + a_inc]) : 0.0;
    f.d(g.r1[i] * c::a0).d(I).nl();
  }
}

}  // namespace uppe
