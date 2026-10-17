#include "blas.hpp"

#if defined(UPPE_USE_MKL)
#include <mkl.h>
#else
#include <cblas.h>
#endif

namespace uppe {

void blas_init() {
#if defined(UPPE_USE_MKL)
  mkl_set_num_threads(1);
#elif defined(UPPE_USE_OPENBLAS)
  openblas_set_num_threads(1);
#endif
}

void dgemm_tn(int M, int N, int K, const double* A, int lda, const double* B, int ldb, double* C, int ldc) {
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, 1.0, A, lda, B, ldb, 0.0, C, ldc);
}

}  // namespace uppe
