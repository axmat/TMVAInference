//Code generated automatically by TMVA for Inference of Model file [Sqrt.onnx] at [Mon Sep 19 12:34:37 2022] 

#ifndef TMVA_SOFIE_SQRT
#define TMVA_SOFIE_SQRT

#include<cmath>
#include<vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_Sqrt{
struct Session {
std::vector<float> fTensor_Y = std::vector<float>(6);
float * tensor_Y = fTensor_Y.data();


Session(std::string filename ="") {
   if (filename.empty()) filename = "Sqrt.dat";
}

std::vector<float> infer(float* tensor_X){
   
//---- OperatorSqrt----------------
   for (size_t i = 0; i < 6; i++) {
      tensor_Y[i] = std::sqrt(tensor_X[i]);
   }
   std::vector<float> ret (tensor_Y, tensor_Y + 6);
   return ret;
}
};
} //TMVA_SOFIE_Sqrt

#endif  // TMVA_SOFIE_SQRT
