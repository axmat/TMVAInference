void SofieUnaryOperators() {
  using namespace TMVA::Experimental::SOFIE;
  RModelParser_ONNX parser;

  RModel model = parser.Parse("./Sqrt.onnx");
  model.Generate();
  model.OutputGenerated("Sqrt.hxx");

  RModel model2 = parser.Parse("./Reciprocal.onnx");
  model2.Generate();
  model2.OutputGenerated("Reciprocal.hxx");
}
