#pragma once

#include "common.hpp"
#include "grids.hpp"

namespace uppe {

// LinThread for one band: E(w,kr) *= exp(-kz*dz) * exp(-(kr/krmax)^20) for bins q >= qStart.
// E is [nr][nq] on the non-negative frequencies; bins below qStart are left untouched.
void linear_step(cplx* E, const Propagator& P, const RadialGrid& g, const Band& b, int qStart, double dz);

// KdivKZ for one band: k(w,kr) *= k_plasma * exp(-(kr/krmax)^20) * exp(-(w/omega_max)^20), all q.
void kdivkz(cplx* K, const Propagator& P, const RadialGrid& g, const Band& b);

}  // namespace uppe
