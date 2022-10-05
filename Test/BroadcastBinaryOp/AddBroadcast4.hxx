//Code generated automatically by TMVA for Inference of Model file [AddBroadcast4.onnx] at [Wed Oct  5 12:32:30 2022] 

#ifndef TMVA_SOFIE_ADDBROADCAST4
#define TMVA_SOFIE_ADDBROADCAST4

#include<vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_AddBroadcast4{
struct Session {
std::vector<float> fTensor_Y = std::vector<float>(8);
float * tensor_Y = fTensor_Y.data();
std::vector<float> fTensor_BroadcastedA = std::vector<float>(8);
float * tensor_BroadcastedA = fTensor_BroadcastedA.data();


Session(std::string filename ="") {
   if (filename.empty()) filename = "AddBroadcast4.dat";
}

std::vector<float> infer(float* tensor_B,float* tensor_A){
   
//------ Add
   // Broadcasting uninitialized tensor A
   {
      float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_A, { 2 , 1 }, { 2 , 4 });
      std::copy(data, data + 8, tensor_BroadcastedA);
      delete[] data;
   }
   for (size_t id = 0; id < 8 ; id++){
      tensor_Y[id] = tensor_BroadcastedA[id] + tensor_B[id] ;
   }
   std::vector<float> ret (tensor_Y, tensor_Y + 8);
   return ret;
}
};
} //TMVA_SOFIE_AddBroadcast4

#endif  // TMVA_SOFIE_ADDBROADCAST4
