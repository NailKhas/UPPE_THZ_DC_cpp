#include "nonlinear.hpp"

#include <algorithm>
#include <stdexcept>

namespace uppe {

NonlinearWorkspace::NonlinearWorkspace(const TimeFreq& tf, int nthreads)
    : fft(make_fft_pool(tf.nt_all, nthreads)), ppl(nthreads), ohm(nthreads), binsP(nthreads), binsL(nthreads) {
  for (int t = 0; t < nthreads; ++t) {
    ppl[t].resize(tf.nt_all);
    ohm[t].resize(tf.nt_all);
    binsP[t].assign(static_cast<std::size_t>(tf.nqT) + tf.nqO, cplx(0.0, 0.0));
    binsL[t].assign(static_cast<std::size_t>(tf.nqT) + tf.nqO, cplx(0.0, 0.0));
  }
}

NonlinearOptWorkspace::NonlinearOptWorkspace(const TimeFreq& tf, int nthreads)
    : fft(make_fft_pool(tf.ntO, nthreads)), ppl(nthreads), ohm(nthreads), binsP(nthreads), binsL(nthreads) {
  for (int t = 0; t < nthreads; ++t) {
    ppl[t].resize(tf.ntO);
    ohm[t].resize(tf.ntO);
    binsP[t].assign(tf.nqO, cplx(0.0, 0.0));
    binsL[t].assign(tf.nqO, cplx(0.0, 0.0));
  }
}

namespace {

// Only these bins of the nt_all spectra are used: all THz bins q < nqT, and the optical bins
// q = ratio*p (k(iit) = kbuf(nt_all/ntO*iit)) outside the zeroed THz window.
// Layout of dst: [0, nqT) THz bins, [nqT, nqT+nqO) optical bins.
void gather_bins(const RealFFT& f, const TimeFreq& tf, cplx* dst) {
  for (int q = 0; q < tf.nqT; ++q) dst[q] = centered_bin(f, q);
  for (int p = 0; p < tf.nqO; ++p) {
    const int q = tf.ratio * p;
    if (q >= tf.nqT) dst[tf.nqT + p] = centered_bin(f, q);
  }
}

// kbuf = (j*P*RKer*w**2 + Ohm*w - RDru*j*Ppl)*dz at the frequency w,
// with the complex products of the Fortran expression written out.
inline cplx kbuf_value(const Physics& ph, double dz, double w, const cplx& P, const cplx& Oh, const cplx& L) {
  const double w2 = w * w;
  const double re = ((-P.imag() * ph.RKer) * w2 + Oh.real() * w) + ph.RDru * L.imag();
  const double im = ((P.real() * ph.RKer) * w2 + Oh.imag() * w) - ph.RDru * L.real();
  return cplx(re * dz, im * dz);
}

// omega(nt_all/2+q) of JefThread_THz.
inline double omega_all(const TimeFreq& tf, int q) { return tf.dw_THz * (tf.nt_all / 2 + q) + tf.wmin; }

// freqs(ntO/2+p) of JefThread.
inline double omega_opt(const TimeFreq& tf, int p) { return tf.dw * (tf.ntO / 2 + p) + tf.wmin; }

struct IonTables {
  const double* PN2 = nullptr;
  const double* PO2 = nullptr;
  int last = 0;
};

IonTables ion_tables(const Physics& ph) {
  if (ph.P_N2.size() != static_cast<std::size_t>(c::ion_table_size) || ph.P_O2.size() != ph.P_N2.size())
    throw std::runtime_error("nonlinear_response: ionization tables are not loaded");
  return IonTables{ph.P_N2.data(), ph.P_O2.data(), c::ion_table_size - 1};
}

// The time loop shared by JefThread and JefThread_THz on N samples of the real field E1:
// Kerr polarization P, ionization, Ohm term and the photocurrent (Heun scheme).
// P overwrites E1 in place (P(iit) depends only on E1(iit)); Ppl ends up holding dJ_dt.
// Returns the final Ne; nclamp counts table lookups clamped to the last entry.
inline double medium_response(double* E1, double* Ppl, double* Ohm, int N, double dt, double DC, double I0,
                              double RefOhm, const IonTables& tab, long long& nclamp) {
  double* P = E1;
  double NeN2 = 0.0, NeO2 = 0.0;
  double VelInt_N2 = 0.0, VelInt_O2 = 0.0;
  double Ne_crt = 0.0;
  {
    const double Ep = E1[0];
    P[0] = Ep * (Ep * Ep);
  }
  Ppl[0] = 0.0;
  Ohm[0] = 0.0;
  for (int it = 1; it < N; ++it) {
    const double E = E1[it];
    const double E_abs2 = E * E;
    P[it] = E * E_abs2 + 3.0 * E_abs2 * DC + 3.0 * E * DC * DC;
    const double NeN2_prv = NeN2;
    const double NeO2_prv = NeO2;
    if (I0 * E_abs2 > 1e11) {
      const double v = std::floor(I0 * E_abs2 * 1e-11);
      int idx;
      if (v > tab.last) {
        idx = tab.last;
        ++nclamp;
      } else {
        idx = static_cast<int>(v);
      }
      VelInt_N2 = VelInt_N2 + tab.PN2[idx] * c::t0 / (10e-15) * dt;
      VelInt_O2 = VelInt_O2 + tab.PO2[idx] * c::t0 / (10e-15) * dt;
      NeN2 = 0.78 * (1.0 - std::exp(-VelInt_N2));
      NeO2 = 0.22 * (1.0 - std::exp(-VelInt_O2));
      const double dNeN2 = NeN2 - NeN2_prv;
      const double dNeO2 = NeO2 - NeO2_prv;
      Ne_crt = NeN2 + NeO2;
      Ohm[it] = RefOhm * (dNeO2 * c::WO2 + dNeN2 * c::WN2) / dt / E;
    } else {
      Ohm[it] = 0.0;
    }
    Ppl[it] = Ne_crt * (E + DC);
  }

  // Ppl(iit) is replaced by dJ_dt(iit) once it is no longer needed.
  const double nu = c::omega_collide;
  double J = 0.0;
  for (int it = 0; it < N - 1; ++it) {
    const double dJ = Ppl[it] - nu * J;
    double JBuff = J + dt * dJ;
    JBuff = Ppl[it + 1] - nu * JBuff;
    J = J + dt * (dJ + JBuff) / 2.0;
    Ppl[it] = dJ;
  }
  Ppl[N - 1] = Ppl[N - 2];
  return Ne_crt;
}

}  // namespace

long long nonlinear_response(const NonlinearInputs& in, NonlinearWorkspace& ws) {
  const TimeFreq& tf = *in.tf;
  const Physics& ph = *in.ph;
  const int N = tf.nt_all;
  const int nqT = tf.nqT;
  const int nqO = tf.nqO;
  const int ntO = tf.ntO;
  const int optStart = N / 2 - ntO / 2;
  const IonTables tab = ion_tables(ph);
  const int nthreads = static_cast<int>(ws.fft.size());
  if (nthreads < max_threads()) throw std::runtime_error("nonlinear_response: workspace has too few threads");
  std::vector<long long> clamped(nthreads, 0);

#pragma omp parallel for schedule(dynamic, 1)
  for (int ir = 0; ir < in.nr; ++ir) {
    const int tid = thread_id();
    RealFFT& fft = *ws.fft[tid];
    double* Ppl = ws.ppl[tid].data();
    double* Ohm = ws.ohm[tid].data();

    // E1(t) = THz field on the nt_all grid plus the optical field in the central ntO window.
    centered_backward(fft, in.thz_r + static_cast<std::size_t>(ir) * nqT, nqT);
    double* E1 = fft.time();
    const double* et = in.et + static_cast<std::size_t>(ir) * ntO;
    for (int t = 0; t < ntO; ++t) E1[optStart + t] += et[t];

    long long nclamp = 0;
    const double Ne_crt = medium_response(E1, Ppl, Ohm, N, tf.dt, in.dc, ph.I0, ph.ROhm, tab, nclamp);
    if (Ne_crt > in.ne_max[ir]) in.ne_max[ir] = Ne_crt;

    cplx* bP = ws.binsP[tid].data();
    cplx* bL = ws.binsL[tid].data();
    fft.forward();  // P is already in fft.time()
    gather_bins(fft, tf, bP);
    std::memcpy(fft.time(), Ppl, sizeof(double) * static_cast<std::size_t>(N));
    fft.forward();
    gather_bins(fft, tf, bL);
    std::memcpy(fft.time(), Ohm, sizeof(double) * static_cast<std::size_t>(N));
    fft.forward();

    // kbuf(nt_all/2) = 0; the THz window of kbuf goes to k_THz and is zeroed before sampling k.
    cplx* kT = in.k_thz + static_cast<std::size_t>(ir) * nqT;
    kT[0] = cplx(0.0, 0.0);
    for (int q = 1; q < nqT; ++q)
      kT[q] = kbuf_value(ph, in.dz, omega_all(tf, q), bP[q], centered_bin(fft, q), bL[q]);
    cplx* kO = in.k_opt + static_cast<std::size_t>(ir) * nqO;
    for (int p = 0; p < nqO; ++p) {
      const int q = tf.ratio * p;
      if (q < nqT) {
        kO[p] = cplx(0.0, 0.0);
      } else {
        kO[p] = kbuf_value(ph, in.dz, omega_all(tf, q), bP[nqT + p], centered_bin(fft, q), bL[nqT + p]);
      }
    }
    clamped[tid] += nclamp;
  }

  long long total = 0;
  for (long long v : clamped) total += v;
  return total;
}

long long nonlinear_response_opt(const NonlinearInputs& in, NonlinearOptWorkspace& ws) {
  const TimeFreq& tf = *in.tf;
  const Physics& ph = *in.ph;
  const int ntO = tf.ntO;
  const int nqO = tf.nqO;
  const IonTables tab = ion_tables(ph);
  const int nthreads = static_cast<int>(ws.fft.size());
  if (nthreads < max_threads()) throw std::runtime_error("nonlinear_response_opt: workspace has too few threads");
  std::vector<long long> clamped(nthreads, 0);

#pragma omp parallel for schedule(dynamic, 1)
  for (int ir = 0; ir < in.nr; ++ir) {
    const int tid = thread_id();
    RealFFT& fft = *ws.fft[tid];
    double* Ppl = ws.ppl[tid].data();
    double* Ohm = ws.ohm[tid].data();

    // E1(t) = Et(t,r): the optical field alone on the ntO grid.
    double* E1 = fft.time();
    std::memcpy(E1, in.et + static_cast<std::size_t>(ir) * ntO, sizeof(double) * static_cast<std::size_t>(ntO));

    long long nclamp = 0;
    const double Ne_crt = medium_response(E1, Ppl, Ohm, ntO, tf.dt, in.dc, ph.I0, ph.ROhm, tab, nclamp);
    if (Ne_crt > in.ne_max[ir]) in.ne_max[ir] = Ne_crt;

    cplx* bP = ws.binsP[tid].data();
    cplx* bL = ws.binsL[tid].data();
    centered_forward(fft, bP, nqO);
    std::memcpy(fft.time(), Ppl, sizeof(double) * static_cast<std::size_t>(ntO));
    centered_forward(fft, bL, nqO);
    std::memcpy(fft.time(), Ohm, sizeof(double) * static_cast<std::size_t>(ntO));
    fft.forward();

    // kbuf(ntO/2) = 0. Unlike JefThread_THz, the bins below the THz window edge are kept.
    cplx* kO = in.k_opt + static_cast<std::size_t>(ir) * nqO;
    kO[0] = cplx(0.0, 0.0);
    for (int p = 1; p < nqO; ++p)
      kO[p] = kbuf_value(ph, in.dz, omega_opt(tf, p), bP[p], centered_bin(fft, p), bL[p]);
    clamped[tid] += nclamp;
  }

  long long total = 0;
  for (long long v : clamped) total += v;
  return total;
}

int nonlinear_rows(const std::vector<double>& maxsq, double Imax, double eps, double I0, int pad) {
  const int nr = static_cast<int>(maxsq.size());
  if (!(eps > 0.0)) return nr;
  const double level = std::min(eps * Imax, 1e8 / I0);
  int last = 0;
  for (int i = 0; i < nr; ++i)
    if (maxsq[i] > level) last = i + 1;
  return std::max(1, std::min(nr, last + std::max(pad, last / 4)));
}

}  // namespace uppe
