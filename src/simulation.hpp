#pragma once

#include <csignal>
#include <filesystem>
#include <optional>

#include "constants.hpp"

namespace uppe {

struct RunConfig {
  Sizes sizes;
  std::optional<double> z_max_cm;  // stop once z*Ldifr >= z_max_cm; empty = run until killed (as Fortran)
  long long max_steps = -1;         // stop after this many z steps in total (resumed runs included); < 0 = no limit
  std::filesystem::path data_dir = ".";  // N2.dat, O2.dat
  std::filesystem::path out_dir = ".";   // res/..., SpSIC.dat, Incident.dat
  std::filesystem::path checkpoint_path;  // written whenever the run stops before z_max; empty = never
  std::filesystem::path resume_path;      // continue from this checkpoint instead of FirstStep; empty = fresh run
  std::optional<double> wall_limit_s;     // stop early enough to stay within this wall time, s
  std::optional<double> min_free_bytes;   // stop once the free space on out_dir's disk drops below this
  std::optional<double> snap_dz_cm;       // uniform snapshot interval, cm; empty = 0.25 cm and 0.1 cm in 19-20.2 cm
  bool linear = false;                    // skip the nonlinear response: linear propagation of the incident field
  const volatile std::sig_atomic_t* stop_flag = nullptr;  // nonzero = stop after the current step
};

enum class RunStatus {
  Finished,  // z_max reached
  Stopped,   // max_steps, wall_limit_s, min_free_bytes or stop_flag; the checkpoint (if requested) is written
};

// Program PARALLEL (PARALLEL.f90).
RunStatus run_simulation(const RunConfig& cfg);

}  // namespace uppe
