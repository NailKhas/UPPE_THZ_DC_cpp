#pragma once

#include <filesystem>
#include <vector>

#include "common.hpp"
#include "constants.hpp"

namespace uppe {

// roots[n-1] is the n-th positive zero of J0 (Fortran bess_root(n)).
std::vector<double> bessel_j0_roots(int count);

// Radial grid of nR-1 points. Index i corresponds to Fortran index i+1.
struct RadialGrid {
  std::vector<double> r1, dr1, kr;
  int size() const { return static_cast<int>(r1.size()); }
};

// The three Fortran grids differ only in the operation order of the same formulas.
enum class RadialKind {
  First,    // PrepareHankel (nR = 16000)
  Optical,  // Prepare_R_opt (nR = nR_opt)
  THz       // PrepareKZ     (nR = nR_THz)
};

RadialGrid make_radial(const std::vector<double>& roots, int nR, int nR_opt, RadialKind kind);

// Frequency axis of one band in the Fortran (centred) indexing 0..nt-1.
struct Band {
  int nt = 0;
  std::vector<double> freqs, refrs, krmax;
  std::vector<float> fr;  // frO / frT: frequency in THz, single precision
  int nq() const { return nt / 2; }
};

Band make_band(int nt, double dw, double wmin);

struct Dispersion {
  double ng = 0, d2kdw2 = 0, vg = 0;
};

// ng, d2kdw2 and vg from the optical band (PrepareKZ_opt).
Dispersion compute_dispersion(const Band& opt, const TimeFreq& tf);

inline double kz_factor() { return 1.0 / (c::cL * c::t0) * c::k0 * (c::a0 * c::a0); }
inline double kplasma_factor() { return c::k0 * (c::a0 * c::a0) / c::cL / c::t0; }

// kz is purely imaginary: kz = i*kappa. k_plasma is real.
struct KzValue {
  double kappa;
  double kplasma;
};

inline KzValue kz_value(double f, double n, double kr, double vg) {
  const double kz2 = sq(f * (1.0 + n)) - sq(kr / c::a0 * c::cL * c::t0);
  if (kz2 > 0.0) {
    const double s = std::sqrt(kz2);
    return {(s - f * vg) * kz_factor(), -1. / s * kplasma_factor()};
  }
  return {0.0, 0.0};
}

// Linear propagator tables on the non-negative frequencies q = 0..nq-1 (Fortran index nt/2+q).
struct Propagator {
  int nr = 0, nq = 0;
  AlignedArray<double> kappa;    // [nr][nq], Im kz
  AlignedArray<double> kplasma;  // [nr][nq]
  std::vector<int> cut;          // rows i >= cut[q] have kr_filter exactly 0
  std::vector<double> wfilter;   // exp(-(w/omega_max)^20)
};

Propagator make_propagator(const Band& band, const RadialGrid& grid, double vg, double omega_max);

// res/*.dat files written by Prepare_R_opt, PrepareKZ_opt and PrepareKZ.
void write_radial_files(const std::filesystem::path& outDir, const RadialGrid& g, const char* nameUm,
                        const char* name, const char* nameKr);

enum class BandKind { Optical, THz };
void write_band_files(const std::filesystem::path& outDir, const Band& band, const Propagator& prop, BandKind kind);
void write_collision_file(const std::filesystem::path& outDir, const TimeFreq& tf);

}  // namespace uppe
