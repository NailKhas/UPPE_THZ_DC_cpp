#pragma once

#include <filesystem>
#include <vector>

#include "constants.hpp"
#include "kw_output.hpp"

namespace uppe {

// Runtime constants set by InitConsts.
struct Physics {
  double I0 = 0;
  double DC_norm = 0;  // 1 kV/cm in field units, where I0*E^2 is the intensity
  double RKer = 0;
  double RDru = 0;
  double ROhm = 0;
  double zself = 0;
  double P_Pcr = 0;
  std::vector<double> P_N2, P_O2;  // ionization rate per 10 fs on an intensity grid of 1e11 W/cm^2
};

Physics make_physics();
void load_ionization_tables(Physics& ph, const std::filesystem::path& dir);

// External DC field (c::dc_profile), kV/cm, at the distance z_cm from the lens.
double dc_field_kV(double z_cm);
// The same field in units where I0*E^2 is the intensity (DC_Field of the Fortran code).
inline double dc_field(const Physics& ph, double z_cm) { return dc_field_kV(z_cm) * ph.DC_norm; }

// write_params: console summary plus res/parameters.dat (kw_z_cm/kw_format: the kw_THz schedule of RunConfig).
void write_params(const Physics& ph, double ng, double d2kdw2, int nthreads, const Sizes& sizes,
                  const TimeFreq& tf, const std::filesystem::path& outDir, const std::vector<double>& kw_z_cm = {},
                  KwFormat kw_format = KwFormat::Binary);

}  // namespace uppe
