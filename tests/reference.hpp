#pragma once

// Literal, slow translations of the Fortran kernels (full complex spectra in Fortran index
// order, naive DFTs with the half swap, dot_product Hankel sums). The optimized kernels are
// checked against these on small grids.

#include <cmath>
#include <complex>
#include <vector>

#include "bessel.hpp"
#include "common.hpp"
#include "constants.hpp"
#include "grids.hpp"
#include "physics.hpp"

namespace ref {

using uppe::cplx;
using Vec = std::vector<cplx>;
namespace c = uppe::c;

inline const cplx J(0.0, 1.0);

// Unnormalized DFT; sign -1 is DftiComputeForward, +1 is DftiComputeBackward.
inline Vec dft(const Vec& x, int sign) {
  const std::size_t N = x.size();
  const double two_pi = 2.0 * std::acos(-1.0);
  std::vector<cplx> tw(N);
  for (std::size_t k = 0; k < N; ++k) tw[k] = std::polar(1.0, sign * two_pi * static_cast<double>(k) / N);
  Vec y(N);
  for (std::size_t k = 0; k < N; ++k) {
    cplx s(0.0, 0.0);
    for (std::size_t n = 0; n < N; ++n) s += x[n] * tw[(k * n) % N];
    y[k] = s;
  }
  return y;
}

// y(0:N/2-1) = x(N/2:N-1); y(N/2:N-1) = x(0:N/2-1)
inline Vec swap_halves(const Vec& x) {
  const std::size_t N = x.size();
  Vec y(N);
  for (std::size_t n = 0; n < N; ++n) y[n] = x[(n + N / 2) % N];
  return y;
}

inline Vec fortran_forward(const Vec& x) { return swap_halves(dft(swap_halves(x), -1)); }

inline Vec fortran_backward(const Vec& x) {
  Vec y = swap_halves(dft(swap_halves(x), +1));
  for (cplx& v : y) v /= static_cast<double>(x.size());
  return y;
}

// Full Fortran-order spectrum (length nt) of a real signal from its stored non-negative half.
inline Vec full_from_half(const cplx* half, int nt) {
  const int nq = nt / 2;
  Vec f(nt, cplx(0.0, 0.0));
  for (int q = 0; q < nq; ++q) f[nq + q] = half[q];
  for (int q = 1; q < nq; ++q) f[nq - q] = std::conj(half[q]);
  return f;
}

// Fortran Y0(k,i) = 2 J0(j_k j_i / j_N) / J1(j_k)^2 and the separate division by j_N.
struct Hankel {
  int n = 0;
  std::vector<double> Y0;
  double jN = 0.0;
};

inline Hankel hankel_matrix(const std::vector<double>& roots, int nR) {
  Hankel H;
  H.n = nR - 1;
  H.jN = roots[nR - 1];
  H.Y0.resize(static_cast<std::size_t>(H.n) * H.n);
  for (int k = 0; k < H.n; ++k)
    for (int i = 0; i < H.n; ++i)
      H.Y0[static_cast<std::size_t>(k) * H.n + i] =
          2.0 * uppe::bessel_j0(roots[k] * roots[i] / H.jN) / std::pow(uppe::bessel_j1(roots[k]), 2);
  return H;
}

// Spectr(iir) = dot_product(YY(:,iir), Origin) / bess_root(nR)
inline Vec hnk(const Hankel& H, const Vec& origin) {
  Vec s(H.n);
  for (int i = 0; i < H.n; ++i) {
    cplx acc(0.0, 0.0);
    for (int k = 0; k < H.n; ++k) acc += H.Y0[static_cast<std::size_t>(k) * H.n + i] * origin[k];
    s[i] = acc / H.jN;
  }
  return s;
}

// kz and k_plasma as complex arrays, layout [iit][ir] (Fortran kz(0:nt-1,1:nR-1)).
struct Kz {
  int nt = 0, nr = 0;
  Vec kz, kp;
  cplx& at(Vec& v, int iit, int ir) { return v[static_cast<std::size_t>(iit) * nr + ir]; }
};

inline Kz kz_tables(const uppe::Band& b, const uppe::RadialGrid& g, double vg) {
  Kz K;
  K.nt = b.nt;
  K.nr = g.size();
  K.kz.assign(static_cast<std::size_t>(K.nt) * K.nr, cplx(0.0, 0.0));
  K.kp.assign(K.kz.size(), cplx(0.0, 0.0));
  const int nt = K.nt;
  for (int ir = 0; ir < K.nr; ++ir) {
    for (int iit = nt / 2 + 1; iit < nt; ++iit) {
      const double kz2 = std::pow(b.freqs[iit] * (1.0 + b.refrs[iit]), 2) - std::pow(g.kr[ir] / c::a0 * c::cL * c::t0, 2);
      if (kz2 > 0.0) {
        K.at(K.kz, iit, ir) = std::sqrt(kz2) - b.freqs[iit] * vg;
        K.at(K.kz, nt - iit, ir) = -K.at(K.kz, iit, ir);
        K.at(K.kp, iit, ir) = -1. / std::sqrt(kz2);
        K.at(K.kp, nt - iit, ir) = K.at(K.kp, iit, ir);
      }
    }
  }
  const cplx kzfac = J / (c::cL * c::t0) * c::k0 * std::pow(c::a0, 2);
  const double kpfac = c::k0 * std::pow(c::a0, 2) / c::cL / c::t0;
  for (cplx& v : K.kz) v *= kzfac;
  for (cplx& v : K.kp) v *= kpfac;
  return K;
}

// LinThread on E(0:nt-1, 1:nr), layout [iit][ir].
inline void lin_thread(Vec& E, Kz& K, const uppe::RadialGrid& g, const uppe::Band& b, int qStart, double dz) {
  const int nt = K.nt, nr = K.nr;
  for (int iit = nt / 2 + qStart; iit < nt; ++iit) {
    for (int ir = 0; ir < nr; ++ir) {
      cplx& e = E[static_cast<std::size_t>(iit) * nr + ir];
      e = e * std::exp(-K.at(K.kz, iit, ir) * dz) * std::exp(-std::pow(g.kr[ir] / b.krmax[iit], 20));
      E[static_cast<std::size_t>(nt - iit) * nr + ir] = std::conj(e);
    }
  }
}

// KdivKZ on k(0:nt-1, 1:nr), layout [iit][ir].
inline void kdivkz(Vec& k, Kz& K, const uppe::RadialGrid& g, const uppe::Band& b, double omega_max) {
  const int nt = K.nt, nr = K.nr;
  for (int iit = nt / 2; iit < nt; ++iit) {
    for (int ir = 0; ir < nr; ++ir) {
      cplx& v = k[static_cast<std::size_t>(iit) * nr + ir];
      v = v * K.at(K.kp, iit, ir) * std::exp(-std::pow(g.kr[ir] / b.krmax[iit], 20)) *
          std::exp(-std::pow(b.freqs[iit] / omega_max, 20));
      k[static_cast<std::size_t>(nt - iit) * nr + ir] = std::conj(v);
    }
  }
  for (int ir = 0; ir < nr; ++ir) k[ir] = cplx(0.0, 0.0);
}

struct JefResult {
  Vec k;      // [ir][ntO]
  Vec k_thz;  // [ir][ntT]
  std::vector<double> ne_max;
  long long clamped = 0;
  long long ionizing_samples = 0;
};

// JefThread_THz. ebuf: [ir][ntT] THz field (w,r); et: [ir][ntO] optical field (t,r).
inline JefResult jef_thread(const uppe::TimeFreq& tf, const uppe::Physics& ph, const Vec& ebuf, const Vec& et,
                            int nr, double dz) {
  const int N = tf.nt_all, ntO = tf.ntO, ntT = tf.ntT;
  const double dt = tf.dt;
  const double DC = ph.DC_Field;
  JefResult R;
  R.k.assign(static_cast<std::size_t>(nr) * ntO, cplx(0.0, 0.0));
  R.k_thz.assign(static_cast<std::size_t>(nr) * ntT, cplx(0.0, 0.0));
  R.ne_max.assign(nr, 0.0);

  std::vector<double> omega(N);
  omega[0] = tf.wmin;
  for (int m = 1; m < N; ++m) omega[m] = tf.dw_THz * m + tf.wmin;

  for (int ir = 0; ir < nr; ++ir) {
    Vec E1(N, cplx(0.0, 0.0));
    for (int iit = 0; iit < ntT; ++iit) E1[N / 2 - ntT / 2 + iit] = ebuf[static_cast<std::size_t>(ir) * ntT + iit];
    E1 = fortran_backward(E1);
    for (int t = 0; t < ntO; ++t) E1[N / 2 - ntO / 2 + t] += et[static_cast<std::size_t>(ir) * ntO + t];

    double Ne_N2 = 0, Ne_O2 = 0, VelInt_N2 = 0, VelInt_O2 = 0, Ne_crt = 0;
    Vec P(N), Ppl(N), Ohm(N), dJ_dt(N), Photocurrent(N);
    cplx E_prv = E1[0];
    P[0] = E_prv * std::pow(std::abs(E_prv), 2);
    Ppl[0] = 0.0;
    Ohm[0] = 0.0;
    for (int iit = 1; iit < N; ++iit) {
      const cplx E_crt = E1[iit];
      const double E_abs2 = std::pow(std::abs(E_crt), 2);
      P[iit] = E_crt * E_abs2 + 3.0 * E_abs2 * DC + 3.0 * E_crt * DC * DC;
      const double Ne_N2_prv = Ne_N2, Ne_O2_prv = Ne_O2;
      if (ph.I0 * E_abs2 > 1e11) {
        ++R.ionizing_samples;
        const double v = std::floor(ph.I0 * E_abs2 * 1e-11);
        int idx;
        if (v > c::ion_table_size - 1) {
          idx = c::ion_table_size - 1;
          ++R.clamped;
        } else {
          idx = static_cast<int>(v);
        }
        VelInt_N2 = VelInt_N2 + ph.P_N2[idx] * c::t0 / (10e-15) * dt;
        VelInt_O2 = VelInt_O2 + ph.P_O2[idx] * c::t0 / (10e-15) * dt;
        Ne_N2 = 0.78 * (1.0 - std::exp(-VelInt_N2));
        Ne_O2 = 0.22 * (1.0 - std::exp(-VelInt_O2));
        const double dNe_N2 = Ne_N2 - Ne_N2_prv;
        const double dNe_O2 = Ne_O2 - Ne_O2_prv;
        Ne_crt = Ne_N2 + Ne_O2;
        Ohm[iit] = ph.ROhm * (dNe_O2 * c::WO2 + dNe_N2 * c::WN2) / dt / std::conj(E_crt);
      } else {
        Ohm[iit] = 0.0;
      }
      Ppl[iit] = Ne_crt * (E_crt + DC);
      E_prv = E_crt;
    }

    dJ_dt[0] = 0.0;
    Photocurrent[0] = 0.0;
    for (int iit = 0; iit <= N - 2; ++iit) {
      dJ_dt[iit] = Ppl[iit] - c::omega_collide * Photocurrent[iit];
      cplx JBuff = Photocurrent[iit] + dt * dJ_dt[iit];
      JBuff = Ppl[iit + 1] - c::omega_collide * JBuff;
      Photocurrent[iit + 1] = Photocurrent[iit] + dt * (dJ_dt[iit] + JBuff) / 2.0;
    }
    dJ_dt[N - 1] = dJ_dt[N - 2];
    if (Ne_crt > R.ne_max[ir]) R.ne_max[ir] = Ne_crt;
    Ppl = dJ_dt;

    P = fortran_forward(P);
    Ppl = fortran_forward(Ppl);
    Ohm = fortran_forward(Ohm);

    Vec kbuf(N);
    for (int m = 0; m < N; ++m)
      kbuf[m] = (J * P[m] * ph.RKer * std::pow(omega[m], 2) + Ohm[m] * omega[m] - ph.RDru * J * Ppl[m]) * dz;
    kbuf[0] = 0.0;
    kbuf[N / 2] = 0.0;
    for (int iit = 0; iit < ntT; ++iit) R.k_thz[static_cast<std::size_t>(ir) * ntT + iit] = kbuf[N / 2 - ntT / 2 + iit];
    for (int m = N / 2 - ntT / 2; m <= N / 2 + ntT / 2 - 1; ++m) kbuf[m] = 0.0;
    for (int iit = 0; iit < ntO; ++iit) R.k[static_cast<std::size_t>(ir) * ntO + iit] = kbuf[N / ntO * iit];
  }
  return R;
}

// FirstStep on the grid g; returns Ebuf(0:nt-1, 1:n) with layout [iit][ir].
inline Vec first_step(const uppe::TimeFreq& tf, const uppe::Band& b, const uppe::RadialGrid& g, const Hankel& H, Kz& K,
                      double dz, double dz0) {
  const int nt = tf.ntO, n = H.n;
  Vec Origin(nt);
  for (int iit = 0; iit < nt; ++iit) {
    const double t = iit * tf.dt + tf.tmin;
    Origin[iit] = std::exp(-.5 * std::pow(t, 2)) * std::cos(c::w0 * c::t0 * t);
  }
  Origin = fortran_forward(Origin);
  double ISm = 0.0;
  for (int iit = nt / 2; iit < nt; ++iit) ISm = std::max(ISm, std::pow(std::abs(Origin[iit]), 2));
  const double foc_ph = .5 * std::pow(c::a0, 2) / (c::focus * c::t0 * c::cL);

  Vec Ebuf(static_cast<std::size_t>(nt) * n, cplx(0.0, 0.0));
  for (int iit = nt / 2 + 1; iit < nt; ++iit) {
    if (!(std::pow(std::abs(Origin[iit]), 2) > ISm * 1e-30)) continue;
    Vec Sp(n);
    for (int i = 0; i < n; ++i) Sp[i] = Origin[iit] * std::exp(-std::pow(g.r1[i] * c::a0, 16) / std::pow(c::a00, 16));
    Sp = hnk(H, Sp);
    for (int i = 0; i < n; ++i)
      Sp[i] = Sp[i] * std::exp(-K.at(K.kz, iit, i) * dz0) * std::exp(-std::pow(g.kr[i] / b.krmax[iit], 20));
    Sp = hnk(H, Sp);
    for (int i = 0; i < n; ++i)
      Sp[i] = Sp[i] * std::exp(-std::pow(g.r1[i] * c::a0, 8)) * std::exp(J * foc_ph * std::pow(g.r1[i], 2) * b.freqs[iit]);
    Sp = hnk(H, Sp);
    for (int i = 0; i < n; ++i)
      Sp[i] = Sp[i] * std::exp(-K.at(K.kz, iit, i) * dz) * std::exp(-std::pow(g.kr[i] / b.krmax[iit], 20));
    Sp = hnk(H, Sp);
    for (int i = 0; i < n; ++i) {
      Ebuf[static_cast<std::size_t>(iit) * n + i] = Sp[i];
      Ebuf[static_cast<std::size_t>(nt - iit) * n + i] = std::conj(Sp[i]);
    }
  }
  return Ebuf;
}

}  // namespace ref
