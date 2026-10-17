#pragma once

#include <filesystem>

#include "common.hpp"
#include "constants.hpp"
#include "grids.hpp"
#include "hankel.hpp"

namespace uppe {

// FirstStep: the incident pulse is apertured, propagated by dz0, focused, propagated by dz,
// all on the large radial grid g (nR = 16000 in production).
// Writes the result Ebuf(w,r) for the first nrKeep radial points into eOut ([nrKeep][ntO/2],
// zero-initialized by the caller), plus SpSIC.dat and Incident.dat in outDir.
void first_step(const TimeFreq& tf, const Band& bandO, const RadialGrid& g, const HankelMatrix& H, double vg,
                double dz, double dz0, int nrKeep, cplx* eOut, const std::filesystem::path& outDir);

}  // namespace uppe
