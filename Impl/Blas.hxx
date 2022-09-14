#ifndef TMVA_EXPERIMENTAL_SOFIE_BLAS
#define TMVA_EXPERIMENTAL_SOFIE_BLAS

namespace TMVA {
namespace Experimental {
namespace SOFIE {
namespace BLAS {

extern "C" void sgemm_(const char *transa, const char *transb, const int *m,
                       const int *n, const int *k, const float *alpha,
                       const float *A, const int *lda, const float *B,
                       const int *ldb, const float *beta, float *C,
                       const int *ldc);

extern "C" void saxpy_(const int *n, const float* alpha, const float* x,
                       const int *incx, float* y, const int* incy);

}
}
}
}

#endif
