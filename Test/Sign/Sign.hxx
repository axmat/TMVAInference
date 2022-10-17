//Code generated automatically by TMVA for Inference of Model file [Sign.onnx] at [Mon Oct 17 09:35:01 2022] 

#ifndef TMVA_SOFIE_SIGN
#define TMVA_SOFIE_SIGN

#include<vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_Sign{
struct Session {
std::vector<float> fTensor_Y = std::vector<float>(5);
float * tensor_Y = fTensor_Y.data();


Session(std::string filename ="") {
   if (filename.empty()) filename = "Sign.dat";
}

std::vector<float> infer(float* tensor_X){
   
//---- Operatorop_0
   for (size_t i = 0; i < 5; i++) {
      if (tensor_X[i] > 0.) {
         tensor_Y[i] = 1.;
      } else if (tensor_X[i] < 0.) {
         tensor_Y[i] = -1.;
      } else {
         tensor_Y[i] = 0.;
      }
   }
   std::vector<float> ret (tensor_Y, tensor_Y + 5);
   return ret;
}
};
} //TMVA_SOFIE_Sign

#endif  // TMVA_SOFIE_SIGN
