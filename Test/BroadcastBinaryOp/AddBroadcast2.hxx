//Code generated automatically by TMVA for Inference of Model file [AddBroadcast2.onnx] at [Wed Oct  5 12:32:30 2022] 

#ifndef TMVA_SOFIE_ADDBROADCAST2
#define TMVA_SOFIE_ADDBROADCAST2

#include<vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_AddBroadcast2{
struct Session {
std::vector<float> fTensor_Y = std::vector<float>(120);
float * tensor_Y = fTensor_Y.data();
std::vector<float> fTensor_BroadcastedA = std::vector<float>(120);
float * tensor_BroadcastedA = fTensor_BroadcastedA.data();


Session(std::string filename ="") {
   if (filename.empty()) filename = "AddBroadcast2.dat";
}

std::vector<float> infer(float* tensor_B,float* tensor_A){
   
//------ Add
   // Broadcasting uninitialized tensor A
   {
      float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_A, { 5 }, { 2 , 3 , 4 , 5 });
      std::copy(data, data + 120, tensor_BroadcastedA);
      delete[] data;
   }
   for (size_t id = 0; id < 120 ; id++){
      tensor_Y[id] = tensor_BroadcastedA[id] + tensor_B[id] ;
   }
   std::vector<float> ret (tensor_Y, tensor_Y + 120);
   return ret;
}
};
} //TMVA_SOFIE_AddBroadcast2

#endif  // TMVA_SOFIE_ADDBROADCAST2
