//Code generated automatically by TMVA for Inference of Model file [RangeFloat.onnx] at [Tue Aug 15 08:54:20 2023] 

#ifndef TMVA_SOFIE_RANGEFLOAT
#define TMVA_SOFIE_RANGEFLOAT

#include<vector>
#include "TMVA/SOFIE_common.hxx"

namespace TMVA_SOFIE_RangeFloat{
struct Session {
std::vector<float> tensor_Y;


Session(std::string = "") {
}

std::vector<float> infer(float* tensor_start,float* tensor_limit,float* tensor_delta){

//------ Range
   size_t op_0_size = static_cast<size_t>(std::max(std::ceil((static_cast<float>(*tensor_limit) - static_cast<float>(*tensor_start)) / static_cast<float>(*tensor_delta)), 0.0f));
   tensor_Y.resize(op_0_size);
   for (size_t i = 0; i < op_0_size ; i++) {
      tensor_Y[i] = *tensor_start + i * (*tensor_delta);
   }
   std::vector<float> ret (tensor_Y);
   return ret;
}
};
} //TMVA_SOFIE_RangeFloat

#endif  // TMVA_SOFIE_RANGEFLOAT
