#pragma once

#include <filesystem>
#include <vector>

#include "constants.hpp"

namespace uppe {

// Runtime constants set by InitConsts.
struct Physics {
  double I0 = 0;
  double DC_Field_in_kV = 0;
  double DC_Field = 0;
  double RKer = 0;
  double RDru = 0;
  double ROhm = 0;
  double zself = 0;
  double P_Pcr = 0;
  std::vector<double> P_N2, P_O2;  // ionization rate per 10 fs on an intensity grid of 1e11 W/cm^2
};

Physics make_physics();
void load_ionization_tables(Physics& ph, const std::filesystem::path& dir);

// write_params: console summary plus res/parameters.dat.
void write_params(const Physics& ph, double ng, double d2kdw2, int nthreads, const Sizes& sizes,
                  const TimeFreq& tf, const std::filesystem::path& outDir);

}  // namespace uppe
