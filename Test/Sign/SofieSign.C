#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"
#include "onnx_proto3.pb.h"

using namespace TMVA::Experimental::SOFIE;

template <typename T> struct ROperator_Sign : public ROperator {
  std::string fNX;
  std::string fNY;
  std::vector<size_t> fShapeX;
  std::vector<size_t> fShapeY;

  ROperator_Sign(std::string nameX, std::string nameY)
      : fNX(nameX), fNY(nameY) {}

  std::vector<std::vector<size_t>>
  ShapeInference(std::vector<std::vector<size_t>> input) override {
    return input;
  }

  std::vector<ETensorType>
  TypeInference(std::vector<ETensorType> input) override {
    return input;
  }

  void Initialize(RModel &model) override {
    if (!model.CheckIfTensorAlreadyExist(fNX)) {
      throw std::runtime_error("TMVA::SOFIE - Tensor " + fNX + " not found.");
    }
    fShapeX = model.GetTensorShape(fNX);
    fShapeY = fShapeX;
    // fShapeY = ShapeInference({fShapeX})[0];
    model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
  }

  std::string Generate(std::string OpName) override {
    OpName = "op_" + OpName;
    std::stringstream out;

    out << SP << "\n//---- Operator" << OpName << "\n";
    size_t length = ConvertShapeToLength(fShapeX);
    out << SP << "for (size_t i = 0; i < " << length << "; i++) {\n";
    out << SP << SP << "if (tensor_" << fNX << "[i] > 0.) {\n";
    out << SP << SP << SP << "tensor_" << fNY << "[i] = 1.;\n";
    out << SP << SP << "} else if (tensor_" << fNX << "[i] < 0.) {\n";
    out << SP << SP << SP << "tensor_" << fNY << "[i] = -1.;\n";
    out << SP << SP << "} else {\n";
    out << SP << SP << SP << "tensor_" << fNY << "[i] = 0.;\n";
    out << SP << SP << "}\n";
    out << SP << "}\n";
    return out.str();
  }
};

// Parser for onnx::NodeProto of type Sign
ParserFuncSignature ParseSign =
    [](RModelParser_ONNX &parser,
       const onnx::NodeProto &nodeproto) -> std::unique_ptr<ROperator> {
  std::unique_ptr<ROperator> op;

  ETensorType input_type = ETensorType::UNDEFINED;
  const std::string input_name = nodeproto.input(0);
  if (parser.IsRegisteredTensorType(input_name)) {
    input_type = parser.GetTensorType(input_name);
  } else {
    throw std::runtime_error(
        "TMVA::SOFIE ONNX Parser Sign op has input tensor " + input_name +
        " but its type is not yet registered");
  }

  const std::string output_name = nodeproto.output(0);
  switch (input_type) {
  case ETensorType::FLOAT:
    op.reset(new ROperator_Sign<float>(input_name, output_name));
    break;
  default:
    throw std::runtime_error("TMVA::SOFIE - Unsupported - Sign Operator does "
                             "not support imput type " +
                             std::to_string(static_cast<int>(input_type)));
  }

  if (!parser.IsRegisteredTensorType(output_name)) {
    parser.RegisterTensorType(output_name, input_type);
  }
  return op;
};

void SofieSign() {

  RModelParser_ONNX parser;

  // Register the Sign operator
  parser.RegisterOperator("Sign", ParseSign);

  // Parse the model
  RModel model = parser.Parse("Sign.onnx");
  model.Generate();
  model.OutputGenerated();
}
