//Code generated automatically by TMVA for Inference of Model file [Reciprocal.onnx] at [Mon Sep 19 12:34:37 2022] 

#ifndef TMVA_SOFIE_RECIPROCAL
#define TMVA_SOFIE_RECIPROCAL

#include<vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_Reciprocal{
struct Session {
std::vector<float> fTensor_Y = std::vector<float>(6);
float * tensor_Y = fTensor_Y.data();


Session(std::string filename ="") {
   if (filename.empty()) filename = "Reciprocal.dat";
}

std::vector<float> infer(float* tensor_X){
   
//---- OperatorReciprocal----------------
   for (size_t i = 0; i < 6; i++) {
      tensor_Y[i] = 1/tensor_X[i];
   }
   std::vector<float> ret (tensor_Y, tensor_Y + 6);
   return ret;
}
};
} //TMVA_SOFIE_Reciprocal

#endif  // TMVA_SOFIE_RECIPROCAL
