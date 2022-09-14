void SofieConvTranspose() {
  using namespace TMVA::Experimental::SOFIE;
  RModelParser_ONNX parser;

  RModel model = parser.Parse("./ConvTranspose1d.onnx");
  model.Generate();
  model.OutputGenerated("ConvTranspose1d.hxx");

  RModel model2 = parser.Parse("./ConvTranspose2d.onnx");
  model2.Generate();
  model2.OutputGenerated("ConvTranspose2d.hxx");

  RModel model3 = parser.Parse("./ConvTranspose3d.onnx");
  model3.Generate();
  model3.OutputGenerated("ConvTranspose3d.hxx");

  RModel model4 = parser.Parse("./ConvTransposeBias2d.onnx");
  model4.Generate();
  model4.OutputGenerated("ConvTransposeBias2d.hxx");

  RModel model5 = parser.Parse("./ConvTransposeGrouped2d.onnx");
  model5.Generate();
  model5.OutputGenerated("ConvTransposeGrouped2d.hxx");
}
