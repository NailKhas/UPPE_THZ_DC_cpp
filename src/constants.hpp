#pragma once

// Parameters of module FME_r (FME_r.f90). Expressions keep the Fortran operation order.
// Fortran default-real literals (single precision) are reproduced as static_cast<double>(...f).

namespace uppe {
namespace c {

// The Fortran value of pi is truncated; every derived grid depends on it, so it is kept.
inline constexpr double pi = 3.1415926536;
inline constexpr double h = 1.055e-34;
inline constexpr double el = -4.8e-10;
inline constexpr double me = 9.1e-28;
inline constexpr double cL = 29979245800.0;
inline constexpr double N0 = 2.7e19;

inline constexpr double tmin_THz = -800.0;
inline constexpr double tmax_THz = 800.0;

inline constexpr double Rmax = 20.0;

// Focal distance, cm; set at configure time with -DUPPE_FOCUS_CM=<cm> (the Fortran value is 20).
#ifndef UPPE_FOCUS_CM
#define UPPE_FOCUS_CM 20.0
#endif
inline constexpr double focus = UPPE_FOCUS_CM;
// Start distance of the propagation (the first step), cm; -DUPPE_ZSIC_CM=<cm> (the Fortran value is 18).
#ifndef UPPE_ZSIC_CM
#define UPPE_ZSIC_CM 18.0
#endif
inline constexpr double zsic = UPPE_ZSIC_CM;
static_assert(zsic < focus, "zsic must be less than focus");
inline constexpr double t1 = 44.0e-15;
inline constexpr double t0 = 54.0e-15;
inline constexpr double t00 = 20.4e-15;
inline constexpr double a00 = 0.1;
inline constexpr double a0 = a00 * (1 - zsic / focus);
inline constexpr double err = static_cast<double>(1e-12f);

inline constexpr double lmbd0 = 744e-7;
inline constexpr double lmbd1 = 400e-7;
inline constexpr double w0 = 2 * pi * cL / lmbd0;
inline constexpr double k0 = 2 * pi / lmbd0;
inline constexpr double Ldifr = k0 * a0 * a0;
inline constexpr double w1 = 2 * pi * cL / lmbd1;

inline constexpr double omega_max = 5 * w0 * t0;
inline constexpr double omega_max_THz = 90.e12 * t0 * 2. * pi;
inline constexpr double alpha_max = pi / 180.0 * 80.0;

inline constexpr double omega_collide = 5e12 * t0;

inline constexpr double n2 = 1.0e-19;

// Medium 1 (air): ionization potentials of N2 and O2, eV.
inline constexpr double WN2 = 15.5;
inline constexpr double WO2 = 12.2;

inline constexpr double WW0 = 0.5e-3;

// InitConsts reads indices 0..9998 of P_N2/P_O2 from N2.dat/O2.dat.
inline constexpr int ion_table_size = 9999;

}  // namespace c

// Grid sizes. The defaults are the Fortran parameters; tests use smaller grids.
struct Sizes {
  int nt_all = 1024 * 16 * 32;
  int nR_first = 16000;  // initial value of nR, used by FirstStep
  int nR_opt = 1000;
  int nR_THz = 5000;
};

struct TimeFreq {
  int nt_all = 0, ntO = 0, ntT = 0;
  int nqO = 0, nqT = 0;  // number of stored non-negative frequencies (ntO/2, ntT/2)
  int ratio = 0;         // nt_all / ntO
  double tmin = 0, tmax = 0, dt = 0, dw = 0, wmax = 0, wmin = 0;
  double dt_THz = 0, dw_THz = 0, wmax_THz = 0, wmin_THz = 0;
  int opt_lin_start = 0;  // int(wmax_THz/dw): first optical bin above the centre propagated by LinThread
};

inline TimeFreq make_time_freq(int nt_all) {
  TimeFreq g;
  g.nt_all = nt_all;
  g.ntO = nt_all / 64;
  g.ntT = nt_all / 32;
  g.nqO = g.ntO / 2;
  g.nqT = g.ntT / 2;
  g.ratio = nt_all / g.ntO;
  g.tmin = c::tmin_THz * g.ntO / nt_all;
  g.tmax = c::tmax_THz * g.ntO / nt_all;
  g.dt = (g.tmax - g.tmin) / g.ntO;
  g.dw = 2 * c::pi / (g.tmax - g.tmin);
  g.wmax = (g.dw * g.ntO / 2 - g.dw);
  g.wmin = -g.dw * g.ntO / 2;
  g.dt_THz = (c::tmax_THz - c::tmin_THz) / g.ntT;
  g.dw_THz = 2 * c::pi / (c::tmax_THz - c::tmin_THz);
  g.wmax_THz = (g.dw_THz * g.ntT / 2 - g.dw_THz);
  g.wmin_THz = -g.dw_THz * g.ntT / 2;
  g.opt_lin_start = static_cast<int>(g.wmax_THz / g.dw);
  return g;
}

}  // namespace uppe
