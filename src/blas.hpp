#pragma once

namespace uppe {

// Forces single-threaded BLAS: parallelism comes from our OpenMP loops around the calls.
void blas_init();

// Row-major C[M x N] = A^T * B, where A is K x M (leading dimension lda >= M)
// and B is K x N (leading dimension ldb >= N).
void dgemm_tn(int M, int N, int K, const double* A, int lda, const double* B, int ldb, double* C, int ldc);

}  // namespace uppe
