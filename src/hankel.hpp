#pragma once

#include <vector>

#include "common.hpp"

namespace uppe {

// Discrete Hankel transform of HnkThread: Spectr(i) = sum_k Y0(k,i) * Origin(k) / bess_root(nR)
// with Y0(k,i) = 2 J0(j_k j_i / j_N) / J1(j_k)^2. The division by j_N is folded into Y.
// The same matrix maps r -> kr and kr -> r.
struct HankelMatrix {
  int n = 0;                 // nR - 1
  AlignedArray<double> Y;    // Y[k*n + i], row-major
};

HankelMatrix make_hankel(const std::vector<double>& roots, int nR);

// Transform input a + beta*b (b optional); both arrays are [rows][nq] complex.
struct HankelInput {
  const cplx* a = nullptr;
  const cplx* b = nullptr;
  double beta = 0.0;
  int rows = 0;
};

// Applies the transform to every frequency column q of a [rows][nq] spectral array (HnkThread).
//   out[i][q] = sum_{k < K} Y[k][i] * in[k][q]   for i < M_q,
//   out[i][q] = 0                                for M_q <= i < M,
//   rows i >= M of out are not touched.
// sumCut (optional): the caller guarantees in[k][q] == 0 for k >= sumCut[q].
// outCut (optional): outputs i >= outCut[q] are known to be multiplied by 0 later; they are set to 0.
// Out-of-place only.
void hankel_spectral(const HankelMatrix& H, const HankelInput& in, cplx* out, int M, int nq, const int* sumCut,
                     const int* outCut);

// out[i][c] = sum_k Y[k][i] * in[k][c] for all n rows and ncols real columns, both row-major with
// leading dimension ncols. Out-of-place. Used by FirstStep, where a few frequencies meet a large n.
void hankel_dense(const HankelMatrix& H, const double* in, double* out, int ncols);

}  // namespace uppe
