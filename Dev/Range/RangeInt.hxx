//Code generated automatically by TMVA for Inference of Model file [RangeInt.onnx] at [Tue Aug 15 08:54:20 2023] 

#ifndef TMVA_SOFIE_RANGEINT
#define TMVA_SOFIE_RANGEINT

#include<vector>
#include "TMVA/SOFIE_common.hxx"

namespace TMVA_SOFIE_RangeInt{
struct Session {
std::vector<int64_t> tensor_Y;


Session(std::string = "") {
}

std::vector<int64_t> infer(int64_t* tensor_start,int64_t* tensor_limit,int64_t* tensor_delta){

//------ Range
   size_t op_0_size = static_cast<size_t>(std::max(std::ceil((static_cast<float>(*tensor_limit) - static_cast<float>(*tensor_start)) / static_cast<float>(*tensor_delta)), 0.0f));
   tensor_Y.resize(op_0_size);
   for (size_t i = 0; i < op_0_size ; i++) {
      tensor_Y[i] = *tensor_start + i * (*tensor_delta);
   }
   std::vector<int64_t> ret (tensor_Y);
   return ret;
}
};
} //TMVA_SOFIE_RangeInt

#endif  // TMVA_SOFIE_RANGEINT
