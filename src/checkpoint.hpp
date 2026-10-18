#pragma once

#include <filesystem>

#include "common.hpp"
#include "constants.hpp"

namespace uppe {

// Everything the main loop of run_simulation carries from one z step to the next,
// apart from the fields E(w,kr) and E_THz(w,kr).
struct LoopState {
  long long iz = 0;
  long long clamped_total = 0;
  double z = 0.0;
  double z_flag = 0.0;
  double Imax = 0.0;
};

// Binary dump of the loop state and both fields. The file is written as <file>.tmp and then
// renamed, so a write that is cut short never replaces a good checkpoint.
void write_checkpoint(const std::filesystem::path& file, const Sizes& sizes, const LoopState& state,
                      const AlignedArray<cplx>& E_O, const AlignedArray<cplx>& E_T);

// E_O and E_T must already have their production sizes. Throws if the file is damaged
// or was written for other grid sizes or another focus or zsic.
LoopState read_checkpoint(const std::filesystem::path& file, const Sizes& sizes, AlignedArray<cplx>& E_O,
                          AlignedArray<cplx>& E_T);

}  // namespace uppe
