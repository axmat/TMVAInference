void SofieConvTranspose() {
  using namespace TMVA::Experimental::SOFIE;
  RModelParser_ONNX parser;

  RModel model4 = parser.Parse("./ConvTransposeBias2d.onnx");
  model4.Generate(Options::kNoSession | Options::kNoWeightFile);
  model4.OutputGenerated("ConvTransposeBias2d.hxx");

}
