#pragma once

#include <filesystem>
#include <optional>

#include "constants.hpp"

namespace uppe {

struct RunConfig {
  Sizes sizes;
  std::optional<double> z_max_cm;  // stop once z*Ldifr >= z_max_cm; empty = run until killed (as Fortran)
  long long max_steps = -1;         // stop after this many z steps; < 0 = no limit
  std::filesystem::path data_dir = ".";  // N2.dat, O2.dat
  std::filesystem::path out_dir = ".";   // res/..., SpSIC.dat, Incident.dat
};

// Program PARALLEL (PARALLEL.f90).
void run_simulation(const RunConfig& cfg);

}  // namespace uppe
