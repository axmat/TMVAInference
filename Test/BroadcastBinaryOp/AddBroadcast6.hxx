//Code generated automatically by TMVA for Inference of Model file [AddBroadcast6.onnx] at [Wed Oct  5 12:32:30 2022] 

#ifndef TMVA_SOFIE_ADDBROADCAST6
#define TMVA_SOFIE_ADDBROADCAST6

#include<vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_AddBroadcast6{
struct Session {
std::vector<float> fTensor_Y = std::vector<float>(48);
float * tensor_Y = fTensor_Y.data();
std::vector<float> fTensor_BroadcastedA = std::vector<float>(48);
float * tensor_BroadcastedA = fTensor_BroadcastedA.data();


Session(std::string filename ="") {
   if (filename.empty()) filename = "AddBroadcast6.dat";
}

std::vector<float> infer(float* tensor_B,float* tensor_A){
   
//------ Add
   // Broadcasting uninitialized tensor A
   {
      float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_A, { 2 , 1 , 3 , 1 , 2 }, { 2 , 2 , 3 , 2 , 2 });
      std::copy(data, data + 48, tensor_BroadcastedA);
      delete[] data;
   }
   for (size_t id = 0; id < 48 ; id++){
      tensor_Y[id] = tensor_BroadcastedA[id] + tensor_B[id] ;
   }
   std::vector<float> ret (tensor_Y, tensor_Y + 48);
   return ret;
}
};
} //TMVA_SOFIE_AddBroadcast6

#endif  // TMVA_SOFIE_ADDBROADCAST6
