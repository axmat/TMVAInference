//Code generated automatically by TMVA for Inference of Model file [LayerNormalization2d.onnx] at [Fri Oct 14 13:15:54 2022] 

#ifndef TMVA_SOFIE_LAYERNORMALIZATION2D
#define TMVA_SOFIE_LAYERNORMALIZATION2D

#include<cmath>
#include<vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_LayerNormalization2d{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
}//BLAS
struct Session {
std::vector<float> fTensor_B = std::vector<float>(4);
float * tensor_B = fTensor_B.data();
std::vector<float> fTensor_Scale = std::vector<float>(4);
float * tensor_Scale = fTensor_Scale.data();
std::vector<float> fTensor_BroadcastedB = std::vector<float>(12);
float * tensor_BroadcastedB = fTensor_BroadcastedB.data();
std::vector<float> fTensor_InvStdDev = std::vector<float>(3);
float * tensor_InvStdDev = fTensor_InvStdDev.data();
std::vector<float> fTensor_Mean = std::vector<float>(3);
float * tensor_Mean = fTensor_Mean.data();
std::vector<float> fTensor_Y = std::vector<float>(12);
float * tensor_Y = fTensor_Y.data();


Session(std::string filename ="") {
   if (filename.empty()) filename = "LayerNormalization2d.dat";
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
   if (length != 4) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 4 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
    for (int i =0; i < length; ++i) 
       f >> tensor_B[i];
   f >> tensor_name >> length;
   if (tensor_name != "tensor_Scale" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_Scale , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 4) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 4 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
    for (int i =0; i < length; ++i) 
       f >> tensor_Scale[i];
   f.close();
   // Broadcasting the bias of LayerNormlization op
   {
      float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_B, { 4 }, { 3 , 4 });
   std::copy(data, data + 12, tensor_BroadcastedB);
   delete[] data;
   }
}

std::vector<float> infer(float* tensor_X){
   // Operator op_0
   std::vector<size_t> op_0_InputShape ({3,4});
   // Compute the mean
   for (size_t axis_0 = 0; axis_0 < op_0_InputShape[0]; axis_0++) {
      float sum = 0.;
      for (size_t axis_1 = 0; axis_1 < op_0_InputShape[1]; axis_1++) {
         sum += tensor_X[axis_0 * 4 + axis_1 * 1];
      }
      tensor_Mean[axis_0 * 1] = sum / float(4);
   }
   // Compute the inverse Standard Deviation
   for (size_t axis_0 = 0; axis_0 < op_0_InputShape[0]; axis_0++){
      float sum = 0.;
      for (size_t axis_1 = 0; axis_1 < op_0_InputShape[1]; axis_1++){
         sum += std::pow(tensor_X[axis_0 * 4 + axis_1 * 1] - tensor_Mean[axis_0 * 1], 2);
      }
      tensor_InvStdDev[axis_0 * 1] = 1 / std::sqrt(sum / float(4) + 1e-05);
   }
   // Y = Scale o InvStdDev (X - Mean)
   for (size_t axis_0 = 0; axis_0 < op_0_InputShape[0]; axis_0++){
      for (size_t axis_1 = 0; axis_1 < op_0_InputShape[1]; axis_1++){
         tensor_Y[axis_0 * 4 + axis_1 * 1] = tensor_Scale[axis_1 * 1] * tensor_InvStdDev[axis_0 * 1] * (tensor_X[axis_0 * 4 + axis_1 * 1] - tensor_Mean[axis_0 * 1]);
      }
   }
   // Add the bias to Y
   int op_0_n = 12;
   float op_0_alpha = 1.;
   int op_0_inc = 1;
   BLAS::saxpy_(&op_0_n, &op_0_alpha, tensor_BroadcastedB, &op_0_inc, tensor_Y, &op_0_inc);
   std::vector<float> ret (tensor_Y, tensor_Y + 12);
   return ret;
}
};
} //TMVA_SOFIE_LayerNormalization2d

#endif  // TMVA_SOFIE_LAYERNORMALIZATION2D
