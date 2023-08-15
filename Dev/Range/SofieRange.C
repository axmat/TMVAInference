#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"

using namespace TMVA::Experimental::SOFIE;

void SofieRange() {
   RModelParser_ONNX parser;
   RModel model = parser.Parse("RangeFloat.onnx");
   model.Generate();
   model.OutputGenerated();
}
