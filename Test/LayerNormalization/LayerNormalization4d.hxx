//Code generated automatically by TMVA for Inference of Model file [LayerNormalization4d.onnx] at [Fri Oct 14 13:15:54 2022] 

#ifndef TMVA_SOFIE_LAYERNORMALIZATION4D
#define TMVA_SOFIE_LAYERNORMALIZATION4D

#include<cmath>
#include<vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_LayerNormalization4d{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
}//BLAS
struct Session {
std::vector<float> fTensor_B = std::vector<float>(20);
float * tensor_B = fTensor_B.data();
std::vector<float> fTensor_Scale = std::vector<float>(20);
float * tensor_Scale = fTensor_Scale.data();
std::vector<float> fTensor_BroadcastedB = std::vector<float>(120);
float * tensor_BroadcastedB = fTensor_BroadcastedB.data();
std::vector<float> fTensor_InvStdDev = std::vector<float>(6);
float * tensor_InvStdDev = fTensor_InvStdDev.data();
std::vector<float> fTensor_Mean = std::vector<float>(6);
float * tensor_Mean = fTensor_Mean.data();
std::vector<float> fTensor_Y = std::vector<float>(120);
float * tensor_Y = fTensor_Y.data();


Session(std::string filename ="") {
   if (filename.empty()) filename = "LayerNormalization4d.dat";
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()){
      throw std::runtime_error("tmva-sofie failed to open file for input weights");
   }
   std::string tensor_name;
   int length;
   f >> tensor_name >> length;
   if (tensor_name != "tensor_B" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_B , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 20) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 20 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
    for (int i =0; i < length; ++i) 
       f >> tensor_B[i];
   f >> tensor_name >> length;
   if (tensor_name != "tensor_Scale" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_Scale , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 20) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 20 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
    for (int i =0; i < length; ++i) 
       f >> tensor_Scale[i];
   f.close();
   // Broadcasting the bias of LayerNormlization op
   {
      float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_B, { 4 , 5 }, { 2 , 3 , 4 , 5 });
   std::copy(data, data + 120, tensor_BroadcastedB);
   delete[] data;
   }
}

std::vector<float> infer(float* tensor_X){
   // Operator op_0
   std::vector<size_t> op_0_InputShape ({2,3,4,5});
   // Compute the mean
   for (size_t axis_0 = 0; axis_0 < op_0_InputShape[0]; axis_0++) {
   for (size_t axis_1 = 0; axis_1 < op_0_InputShape[1]; axis_1++) {
      float sum = 0.;
      for (size_t axis_2 = 0; axis_2 < op_0_InputShape[2]; axis_2++) {
      for (size_t axis_3 = 0; axis_3 < op_0_InputShape[3]; axis_3++) {
         sum += tensor_X[axis_0 * 60 + axis_1 * 20 + axis_2 * 5 + axis_3 * 1];
      }
      }
      tensor_Mean[axis_0 * 3 + axis_1 * 1] = sum / float(20);
   }
   }
   // Compute the inverse Standard Deviation
   for (size_t axis_0 = 0; axis_0 < op_0_InputShape[0]; axis_0++){
   for (size_t axis_1 = 0; axis_1 < op_0_InputShape[1]; axis_1++){
      float sum = 0.;
      for (size_t axis_2 = 0; axis_2 < op_0_InputShape[2]; axis_2++){
      for (size_t axis_3 = 0; axis_3 < op_0_InputShape[3]; axis_3++){
         sum += std::pow(tensor_X[axis_0 * 60 + axis_1 * 20 + axis_2 * 5 + axis_3 * 1] - tensor_Mean[axis_0 * 3 + axis_1 * 1], 2);
      }
      }
      tensor_InvStdDev[axis_0 * 3 + axis_1 * 1] = 1 / std::sqrt(sum / float(20) + 1e-05);
   }
   }
   // Y = Scale o InvStdDev (X - Mean)
   for (size_t axis_0 = 0; axis_0 < op_0_InputShape[0]; axis_0++){
   for (size_t axis_1 = 0; axis_1 < op_0_InputShape[1]; axis_1++){
      for (size_t axis_2 = 0; axis_2 < op_0_InputShape[2]; axis_2++){
      for (size_t axis_3 = 0; axis_3 < op_0_InputShape[3]; axis_3++){
         tensor_Y[axis_0 * 60 + axis_1 * 20 + axis_2 * 5 + axis_3 * 1] = tensor_Scale[axis_2 * 5 + axis_3 * 1] * tensor_InvStdDev[axis_0 * 3 + axis_1 * 1] * (tensor_X[axis_0 * 60 + axis_1 * 20 + axis_2 * 5 + axis_3 * 1] - tensor_Mean[axis_0 * 3 + axis_1 * 1]);
      }
      }
   }
   }
   // Add the bias to Y
   int op_0_n = 120;
   float op_0_alpha = 1.;
   int op_0_inc = 1;
   BLAS::saxpy_(&op_0_n, &op_0_alpha, tensor_BroadcastedB, &op_0_inc, tensor_Y, &op_0_inc);
   std::vector<float> ret (tensor_Y, tensor_Y + 120);
   return ret;
}
};
} //TMVA_SOFIE_LayerNormalization4d

#endif  // TMVA_SOFIE_LAYERNORMALIZATION4D
