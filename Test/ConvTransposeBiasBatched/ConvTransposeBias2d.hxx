//Code generated automatically by TMVA for Inference of Model file [ConvTransposeBias2d.onnx] at [Fri Sep 16 16:31:00 2022] 

#ifndef TMVA_SOFIE_CONVTRANSPOSEBIAS2D
#define TMVA_SOFIE_CONVTRANSPOSEBIAS2D

#include<vector>
#include "TMVA/SOFIE_common.hxx"

namespace TMVA_SOFIE_ConvTransposeBias2d{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
float tensor_B[100] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
float tensor_W[18] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
std::vector<float> fTensor_Y = std::vector<float>(100);
float * tensor_Y = fTensor_Y.data();
std::vector<float> infer(float* tensor_X){

//----  operator ConvTranspose op_0
   float op_0_f[18] = {0};
   for (std::size_t oc = 0; oc < 1; oc++) {
      for (std::size_t ic = 0; ic < 2; ic++) {
         for (std::size_t kh = 0; kh < 3; kh++) {
            for (std::size_t kw = 0; kw < 3; kw++) {
               op_0_f[oc * 18 + ic * 9 + kh * 3 + kw * 1  ] = tensor_W[oc * 18 + ic * 9 + kh * 3 + kw ];
            }
         }
      }
   }
   char op_0_transA = 'N';
   char op_0_transB = 'N';
   int op_0_m = 25;
   int op_0_n = 2;
   int op_0_k = 9;
   float op_0_alpha = 1.0;
   float op_0_beta = 0.0;
   float op_0_xcol[225] = {0};
   for (size_t n = 0; n < 2; n++) {
      size_t x_offset = n * 9;
      size_t out_offset = n * 50;
      TMVA::Experimental::SOFIE::UTILITY::Im2col<float>(tensor_X + x_offset,1,3,3,3,3,2,2,1,1,1,1,op_0_xcol);

       BLAS::sgemm_(&op_0_transA, &op_0_transB, &op_0_m, &op_0_n, &op_0_k, &op_0_alpha, op_0_xcol, &op_0_m,
         op_0_f, &op_0_k, &op_0_beta, tensor_Y + out_offset, &op_0_m);
   }
   int op_0_size = 100;
   float op_0_gamma = 1.0;
   int op_0_incx = 1;
   int op_0_incy = 1;
   BLAS::saxpy_(&op_0_size, &op_0_gamma, tensor_B, &op_0_incx, tensor_Y, &op_0_incy);
   std::vector<float> ret (tensor_Y, tensor_Y + 100);
   return ret;
}
} //TMVA_SOFIE_ConvTransposeBias2d

#endif  // TMVA_SOFIE_CONVTRANSPOSEBIAS2D
