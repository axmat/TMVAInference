//Code generated automatically by TMVA for Inference of Model file [ConvTransposeM.onnx] at [Thu Oct 27 11:59:08 2022] 

#ifndef TMVA_SOFIE_CONVTRANSPOSEM
#define TMVA_SOFIE_CONVTRANSPOSEM

#include<vector>
#include "TMVA/SOFIE_common.hxx"

namespace TMVA_SOFIE_ConvTransposeM{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
}//BLAS
struct Session {
float tensor_B[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
float tensor_W[243] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9};
std::vector<float> fTensor_BroadcastedB = std::vector<float>(9216);
float * tensor_BroadcastedB = fTensor_BroadcastedB.data();
std::vector<float> fTensor_Y = std::vector<float>(36864);
float * tensor_Y = fTensor_Y.data();
std::vector<float> fTensor_TransposeKernelW = std::vector<float>(24883200);
float * tensor_TransposeKernelW = fTensor_TransposeKernelW.data();


Session() {
   {
      float * data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_B, { 9 , 1 , 1 }, { 9 , 32 , 32 });
      std::copy(data, data + 9216, tensor_BroadcastedB);
      delete[] data;
   }
}

std::vector<float> infer(float* tensor_X){

//----  operator ConvTranspose op_0
   for (size_t row = 0; row < 2700; row+=900) {
      for (size_t col = 0; col < 9216; col+=1024) {
         size_t c = row / 900;
         size_t m = col / 1024;
            size_t offset = row * 9216 + col;
            for (size_t h = 0; h < 30; h++) {
               size_t idx = offset + h * 276480 + h * 32;
               for (size_t w = 0; w < 30; w++) {
                  for (size_t kh = 0; kh < 3; kh++) {
                     size_t f_idx = idx + w * 9216 + w + kh * 32;
                     size_t k_offset = c * 81 + m * 9 + kh * 3;
                     std::copy(tensor_W + k_offset, tensor_W + k_offset + 3, tensor_TransposeKernelW + f_idx);
                  }
               }
            }
      }
   }
   for (size_t n = 0; n < 4; n++) {
      char op_0_transA = 'N';
      int op_0_m = 9216;
      int op_0_n = 2700;
      int op_0_inc = 1;
      float op_0_alpha = 1.0;
      float op_0_beta = 0.0;
      size_t in_offset = n * 2700;
      size_t out_offset = n * 9216;
      BLAS::sgemv_(&op_0_transA, &op_0_m, &op_0_n, &op_0_alpha, tensor_TransposeKernelW, &op_0_m,tensor_X + in_offset, &op_0_inc, &op_0_beta, tensor_Y + out_offset, &op_0_inc);
      int op_0_size = 9216;
      float op_0_gamma = 1.0;
      BLAS::saxpy_(&op_0_size, &op_0_gamma, tensor_BroadcastedB, &op_0_inc, tensor_Y + out_offset, &op_0_inc);
   }
   std::vector<float> ret (tensor_Y, tensor_Y + 36864);
   return ret;
}
};
} //TMVA_SOFIE_ConvTransposeM

#endif  // TMVA_SOFIE_CONVTRANSPOSEM
