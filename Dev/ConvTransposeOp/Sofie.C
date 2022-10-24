#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"

#include "ConvTransposeOp.hxx"
#include "ParseConvTransposeOp.hxx"

using namespace TMVA::Experimental::SOFIE;

void Sofie() {

  RModelParser_ONNX parser;

	// REplace ConvTranspose in ROOT master with this version
	parser.RegisterOperator("ConvTranspose", ParseConvTransposeOp);

  RModel model = parser.Parse("./Conv2DTranspose_Relu_Sigmoid.onnx");
  model.Generate(Options::kNoWeightFile);

  model.OutputGenerated();
}
