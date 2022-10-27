#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"

#include "ConvTransposeOp.hxx"
#include "ParseConvTransposeOp.hxx"

using namespace TMVA::Experimental::SOFIE;

void SofieGenConvTranspose() {

  RModelParser_ONNX parser;

   // Replace ConvTranspose in ROOT master with this version
   parser.RegisterOperator("ConvTranspose", ParseConvTransposeOp);

  RModel model = parser.Parse("./ConvTransposeM.onnx");
  model.Generate(Options::kNoWeightFile);

  model.OutputGenerated();
}
