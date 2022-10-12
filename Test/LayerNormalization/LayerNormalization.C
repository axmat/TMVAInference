#include "LayerNormalizationOperator.hxx"

void LayerNormalization() {
  RModelParser_ONNX parser;
  parser.RegisterOperator("LayerNormalization", ParseLayerNormalisation);

  for (auto &filename :
       {"LayerNormalization2d.onnx", "LayerNormalization4d.onnx"}) {
    RModel model = parser.Parse(filename);
    model.Generate();
    model.OutputGenerated();
  }
}
