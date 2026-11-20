#pragma once

#include <vector>

#include "common.hpp"
#include "constants.hpp"
#include "fft.hpp"
#include "physics.hpp"

namespace uppe {

// Per-thread buffers for nonlinear_response; build once, reuse every call.
struct NonlinearWorkspace {
  NonlinearWorkspace(const TimeFreq& tf, int nthreads);
  FftPool fft;                               // size nt_all
  std::vector<AlignedArray<double>> ppl;     // Ppl, then dJ/dt
  std::vector<AlignedArray<double>> ohm;
  std::vector<std::vector<cplx>> binsP, binsL;  // needed bins of FFT(P) and FFT(dJ/dt)
};

// Per-thread buffers for nonlinear_response_opt (the ntO grid).
struct NonlinearOptWorkspace {
  NonlinearOptWorkspace(const TimeFreq& tf, int nthreads);
  FftPool fft;                               // size ntO
  std::vector<AlignedArray<double>> ppl;     // Ppl, then dJ/dt
  std::vector<AlignedArray<double>> ohm;
  std::vector<std::vector<cplx>> binsP, binsL;  // FFT(P) and FFT(dJ/dt), non-negative frequencies
};

struct NonlinearInputs {
  const TimeFreq* tf = nullptr;
  const Physics* ph = nullptr;
  const cplx* thz_r = nullptr;  // THz field E_THz(w,r), [>= nr rows][nqT]; unused by nonlinear_response_opt
  const double* et = nullptr;   // optical field Et(t,r), [nr][ntO]
  int nr = 0;                   // nR_opt - 1
  double dz = 0.0;
  double dc = 0.0;              // external DC field at the z of this RK stage, dc_field(ph, z_cm)
  cplx* k_opt = nullptr;        // out: k(w,r), [nr][nqO]
  cplx* k_thz = nullptr;        // out: k_THz(w,r), [nr][nqT]; unused by nonlinear_response_opt
  double* ne_max = nullptr;     // in/out: Ne_max(r), updated with max
};

// JefThread_THz. Returns how many ionization-table lookups were clamped to the last table entry
// (the Fortran code reads past the end of P_N2/P_O2 in that case).
long long nonlinear_response(const NonlinearInputs& in, NonlinearWorkspace& ws);

// JefThread: the optical field alone on the ntO grid, only k(w,r) is computed.
// Returns the number of clamped ionization-table lookups, as nonlinear_response.
long long nonlinear_response_opt(const NonlinearInputs& in, NonlinearOptWorkspace& ws);

}  // namespace uppe
