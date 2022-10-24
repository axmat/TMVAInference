#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"

using namespace TMVA::Experimental::SOFIE;

void SofieNary() {
   for (auto& op: {"Max", "Min", "Mean", "Sum"}) {
     RModelParser_ONNX parser;
     RModel model = parser.Parse(std::string(op) + "MultidirectionalBroadcast.onnx");
     model.Generate();
      model.OutputGenerated();
   }
}
