//Code generated automatically by TMVA for Inference of Model file [Add_broadcast_2d.onnx] at [Tue Oct  4 10:59:07 2022] 

#ifndef TMVA_SOFIE_ADD_BROADCAST_2D
#define TMVA_SOFIE_ADD_BROADCAST_2D

#include<vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_Add_broadcast_2d{
struct Session {
std::vector<float> fTensor_Y = std::vector<float>(20);
float * tensor_Y = fTensor_Y.data();
std::vector<float> fTensor_BroadcastedA = std::vector<float>(20);
float * tensor_BroadcastedA = fTensor_BroadcastedA.data();


Session(std::string filename ="") {
   if (filename.empty()) filename = "Add_broadcast_2d.dat";
}

std::vector<float> infer(float* tensor_B,float* tensor_A){
   
//------ Add
   // Broadcasting uninitialized tensor A
   {
      float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_A, { 5 }, { 4 , 5 });
      std::copy(data, data + 20, tensor_BroadcastedA);
      delete[] data;
   }
   for (size_t id = 0; id < 20 ; id++){
      tensor_Y[id] = tensor_BroadcastedA[id] + tensor_B[id] ;
   }
   std::vector<float> ret (tensor_Y, tensor_Y + 20);
   return ret;
}
};
} //TMVA_SOFIE_Add_broadcast_2d

#endif  // TMVA_SOFIE_ADD_BROADCAST_2D
