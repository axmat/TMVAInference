#ifndef TMVA_EXPERIMENTAL_SOFIE_BLAS
#define TMVA_EXPERIMENTAL_SOFIE_BLAS

namespace TMVA{
namespace Experimental{
namespace SOFIE{
namespace Blas{

extern "C" void sgemm_(const char *transa, const char *transb, const int *m,
                       const int *n, const int *k, const float *alpha,
                       const float *A, const int *lda, const float *B,
                       const int *ldb, const float *beta, float *C,
                       const int *ldc);

extern "C" void dgemm_(const char *transa, const char *transb, const int *m,
                       const int *n, const int *k, const double *alpha,
                       const double *A, const int *lda, const double *B,
                       const int *ldb, const double *beta, double *C,
                       const int *ldc);



template<typename T>
void Gemm(const char *transa, const char *transb, const int *m, const int *n,
          const int *k, const T *alpha, const T *X, const int *lda, const T *B,
          const int *ldb, const T *beta, T *C, const int *ldc);

template<>
void Gemm<float>(const char *transa, const char *transb, const int *m, const int *n,
                 const int *k, const float *alpha, const float *X, const int *lda, const float *B,
                 const int *ldb, const float *beta, float *C, const int *ldc)
{
   sgemm_(transa, transb, m, n, k, alpha, X, lda, B, ldb, beta, C, ldc);
}

template<>
void Gemm<double>(const char *transa, const char *transb, const int *m, const int *n,
                 const int *k, const double *alpha, const double *X, const int *lda, const double *B,
                 const int *ldb, const double *beta, double *C, const int *ldc)
{
   dgemm_(transa, transb, m, n, k, alpha, X, lda, B, ldb, beta, C, ldc);
}


}
}
}
}

#endif
