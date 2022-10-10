//Code generated automatically by TMVA for Inference of Model file [ConvTranspose1d.onnx] at [Fri Oct  7 10:54:08 2022] 

#ifndef TMVA_SOFIE_CONVTRANSPOSE1D
#define TMVA_SOFIE_CONVTRANSPOSE1D

#include<vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_ConvTranspose1d{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
struct Session {
std::vector<float> fTensor_W = std::vector<float>(6);
float * tensor_W = fTensor_W.data();
std::vector<float> fTensor_Y = std::vector<float>(10);
float * tensor_Y = fTensor_Y.data();

std::vector<float> fVec_op_0_f = std::vector<float>(6);
std::vector<float> fVec_op_0_xcol = std::vector<float>(15);


Session(std::string filename ="") {
   if (filename.empty()) filename = "ConvTranspose1d.dat";
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()){
      throw std::runtime_error("tmva-sofie failed to open file for input weights");
   }
   std::string tensor_name;
   int length;
   f >> tensor_name >> length;
   if (tensor_name != "tensor_W" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_W , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 6) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 6 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
    for (int i =0; i < length; ++i) 
       f >> tensor_W[i];
   f.close();
}

std::vector<float> infer(float* tensor_X){

//----  operator ConvTranspose op_0
   float * op_0_f = fVec_op_0_f.data();
   for (std::size_t oc = 0; oc < 1; oc++) {
      for (std::size_t ic = 0; ic < 2; ic++) {
            for (std::size_t kw = 0; kw < 3; kw++) {
               op_0_f[oc * 6 + ic * 3 + kw * 1  ] = tensor_W[oc * 6 + ic * 3 + kw ];
            }
      }
   }
   char op_0_transA = 'N';
   char op_0_transB = 'N';
   int op_0_m = 5;
   int op_0_n = 2;
   int op_0_k = 3;
   float op_0_alpha = 1.0;
   float op_0_beta = 0.0;
   float * op_0_xcol = fVec_op_0_xcol.data();
   for (size_t n = 0; n < 1; n++) {
      size_t x_offset = n * 3;
      size_t out_offset = n * 10;
      TMVA::Experimental::SOFIE::UTILITY::Im2col<float>(tensor_X + x_offset,1,1,3,1, 3,0,2,1,1,1,1,op_0_xcol);

       BLAS::sgemm_(&op_0_transA, &op_0_transB, &op_0_m, &op_0_n, &op_0_k, &op_0_alpha, op_0_xcol, &op_0_m,
         op_0_f, &op_0_k, &op_0_beta, tensor_Y + out_offset, &op_0_m);
   }
   std::vector<float> ret (tensor_Y, tensor_Y + 10);
   return ret;
}
};
} //TMVA_SOFIE_ConvTranspose1d

#endif  // TMVA_SOFIE_CONVTRANSPOSE1D
