//Code generated automatically by TMVA for Inference of Model file [MaxMultidirectionalBroadcast.onnx] at [Mon Oct 17 15:35:37 2022] 

#ifndef TMVA_SOFIE_MAXMULTIDIRECTIONALBROADCAST
#define TMVA_SOFIE_MAXMULTIDIRECTIONALBROADCAST

#include<cmath>
#include<vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_MaxMultidirectionalBroadcast{
struct Session {
std::vector<float> fTensor_BroadcastedC = std::vector<float>(24);
float * tensor_BroadcastedC = fTensor_BroadcastedC.data();
std::vector<float> fTensor_BroadcastedB = std::vector<float>(24);
float * tensor_BroadcastedB = fTensor_BroadcastedB.data();
std::vector<float> fTensor_BroadcastedA = std::vector<float>(24);
float * tensor_BroadcastedA = fTensor_BroadcastedA.data();
std::vector<float> fTensor_Y = std::vector<float>(24);
float * tensor_Y = fTensor_Y.data();


Session(std::string filename ="") {
   if (filename.empty()) filename = "MaxMultidirectionalBroadcast.dat";
}

std::vector<float> infer(float* tensor_B,float* tensor_C,float* tensor_A){
   
//------ Max operator
      // Broadcasting A to { 2 , 3 , 4 }
      {
         float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_A, { 3 , 1 }, { 2 , 3 , 4 });
         std::copy(data, data + 24, tensor_BroadcastedA);
         delete[] data;
      }
      // Broadcasting B to { 2 , 3 , 4 }
      {
         float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_B, { 2 , 3 , 1 }, { 2 , 3 , 4 });
         std::copy(data, data + 24, tensor_BroadcastedB);
         delete[] data;
      }
      // Broadcasting C to { 2 , 3 , 4 }
      {
         float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_C, { 1 , 4 }, { 2 , 3 , 4 });
         std::copy(data, data + 24, tensor_BroadcastedC);
         delete[] data;
      }
   for (size_t id = 0; id < 24; id++) {
      tensor_Y[id] = tensor_BroadcastedA[id];
      tensor_Y[id] = std::max(tensor_Y[id], tensor_BroadcastedB[id]);
      tensor_Y[id] = std::max(tensor_Y[id], tensor_BroadcastedC[id]);
   }
   std::vector<float> ret (tensor_Y, tensor_Y + 24);
   return ret;
}
};
} //TMVA_SOFIE_MaxMultidirectionalBroadcast

#endif  // TMVA_SOFIE_MAXMULTIDIRECTIONALBROADCAST
