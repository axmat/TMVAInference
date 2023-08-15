#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"

using namespace TMVA::Experimental::SOFIE;

void SofieRange() {
   RModelParser_ONNX parser;

   RModel model = parser.Parse("RangeFloat.onnx");
   model.Generate();
   model.OutputGenerated();

   RModel model2 = parser.Parse("RangeInt.onnx");
   model2.Generate();
   model2.OutputGenerated();
}
