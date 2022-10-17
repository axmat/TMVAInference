#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"

using namespace TMVA::Experimental::SOFIE;

void SofieMax() {

  RModelParser_ONNX parser;
  RModel model = parser.Parse("./MaxMultidirectionalBroadcast.onnx");
  model.Generate();
  model.OutputGenerated();
}
