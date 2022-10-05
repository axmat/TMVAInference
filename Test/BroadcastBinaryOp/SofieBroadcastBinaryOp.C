#include <string>
#include <vector>

void SofieBroadcastBinaryOp() {
   using namespace TMVA::Experimental::SOFIE;
   RModelParser_ONNX parser;

   std::vector<std::string> names(7);
   for (size_t i = 0; i < 7; i++)
      names[i] = "./AddBroadcast" + std::to_string(i + 1) + ".onnx";

   for (auto name : names) {
      RModel model = parser.Parse(name);
      model.Generate();
      model.OutputGenerated();
   }
}
