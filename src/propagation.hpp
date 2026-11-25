#pragma once

#include "common.hpp"
#include "grids.hpp"

namespace uppe {

// LinThread for one band: E(w,kr) *= exp(-kz*dz) * exp(-(kr/krmax)^20) for bins q >= qStart.
// E is [nr][nq] on the non-negative frequencies; bins below qStart are left untouched.
// Rows i >= P.hcut[q] are set to 0: the field there is 0 in the simulation (see Propagator::hcut).
void linear_step(cplx* E, const Propagator& P, const RadialGrid& g, const Band& b, int qStart, double dz);

// KdivKZ for one band: k(w,kr) *= k_plasma * exp(-(kr/krmax)^20) * exp(-(w/omega_max)^20), all q.
// Rows i >= P.hcut[q], where this factor is exactly 0, are set to 0 without reading k.
void kdivkz(cplx* K, const Propagator& P, const RadialGrid& g, const Band& b);

}  // namespace uppe
