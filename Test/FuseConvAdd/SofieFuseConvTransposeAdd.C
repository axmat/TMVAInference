void SofieFuseConvTransposeAdd() {
  using namespace TMVA::Experimental::SOFIE;
  RModelParser_ONNX parser;

  RModel model = parser.Parse("./Conv2DTranspose_Relu_Sigmoid.onnx");
  model.Generate();
  model.OutputGenerated("FuseConvTransposeAdd.hxx");
}
