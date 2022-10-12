#include "TMVA/SOFIE_common.hxx"
#include "onnx_proto3.pb.h"
#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"
#include <sstream>
#include <string>

using namespace TMVA::Experimental::SOFIE;

template<typename T>
class ROperator_LayerNormalization : public ROperator{
private:
   int64_t fAttrAxis;
   float fAttrEpsilon;
   size_t fAttrStashType;

   std::string fNX;
   std::string fNScale;
   std::string fNB;
   std::string fNY;
   std::string fNMean;
   std::string fNInvStdDev;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeScale;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeY;
   std::vector<size_t> fShapeMean;
   std::vector<size_t> fShapeInvStdDev;

   size_t fAxis; // axis in [0, size)
   size_t fSize; // Size of the input
   //size_t fAxisDim;
   size_t fLength; // Lenght of the input X

   std::vector<size_t> fNormalizedShape;
   std::vector<size_t> fAxesShape;
   size_t fNormalizedLength;
   size_t fAxesLength;

   std::string fType;

public:
   ROperator_LayerNormalization(int64_t axis, float epsilon, size_t stashType,
      const std::string& nameX, const std::string& nameScale,
      const std::string& nameB, const std::string& nameY,
      const std::string& nameMean, const std::string& nameInvStdDev):
      fAttrAxis(axis), fAttrEpsilon(epsilon), fAttrStashType(stashType),
      fNX(nameX), fNScale(nameScale), fNB(nameB), fNY(nameY),
      fNMean(nameMean), fNInvStdDev(nameInvStdDev) {}

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      return input;
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::string GenerateInitMemberCode(std::string OpName) {
      std::stringstream out;
      return out.str();
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw
            std::runtime_error("TMVA::SOFIE - Tensor " + fNX + " not found.");
      }
      fShapeX = model.GetTensorShape(fNX);
      fShapeY = fShapeX;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      // Type of the output
      fType = ConvertTypeToString(model.GetTensorType(fNX));
      // Size of the input
      fSize = fShapeX.size();
      // Axis in [0, size)
      fAxis = (fAttrAxis < 0) ? fSize + fAttrAxis : fAttrAxis;
      // Shape of fShapeX[0, ..., fAxis)
      fAxesShape = std::vector<size_t>(fShapeX.begin(), fShapeX.begin() + fAxis);
      // Lenght of the axes
      fAxesLength = ConvertShapeToLength(fAxesShape);
      // Shape of fShapeX[fAxis, ..., fSize)
      fNormalizedShape = std::vector<size_t>(fShapeX.begin() + fAxis, fShapeX.end());
      // Length of the normalized axis
      fNormalizedLength = ConvertShapeToLength(fNormalizedShape);
      // length of the input
      fLength = ConvertShapeToLength(fShapeX);
      // Mean
      //fShapeMean.resize();
       if (fNMean.empty()) {
         fNMean = "Mean";
         model.AddIntermediateTensor(fNMean, model.GetTensorType(fNX), {fAxesLength});
      }
      // Inverse Standard Deviation
       if (fNInvStdDev.empty()) {
         fNInvStdDev = "InvStdDev";
         model.AddIntermediateTensor(fNInvStdDev, model.GetTensorType(fNX), {fAxesLength});
      }
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeX.empty()) {
         throw
            std::runtime_error("TMVA::SOFIE LayerNormalization operator " + OpName + " called to generate without beging initialized first.");
      }
      if (fShapeX.size() > 5) {
         throw
            std::runtime_error("TMVA::SOFIE LayerNormalization operator not implemented for input tensor of size > 5.");
      }

      std::stringstream out;

      out << SP << "// Operator " << OpName << "\n";

      // Loop over all the normalized axes i.e. [axis, ..., size)
      out << SP << "std::vector<size_t> " << OpName << "_InputShape ({";
      for (size_t i = 0; i < fSize; i++) {
         out << fShapeX[i];
         if (i + 1 < fSize) {
            out << ",";
         }
      }
      out << "});\n";
      std::string inputShape = OpName + "_InputShape";

      auto strides = UTILITY::ComputeStrideFromShape(fShapeX);
      std::string InputIndex = "axis_0 * " + std::to_string(strides[0]);
      for (size_t i = 1; i < fSize; i++) {
         InputIndex += " + axis_" + std::to_string(i) + " * " + std::to_string(strides[i]);
      }

      auto axesStrides = UTILITY::ComputeStrideFromShape(fAxesShape);
      std::string axesIndex;
      axesIndex += "axis_" + std::to_string(0) + " * " + std::to_string(axesStrides[0]);
      for (size_t i = 1; i < fAxis; i++) {
         axesIndex += " + axis_" + std::to_string(i) + " * " + std::to_string(axesStrides[i]);
      }

      auto normalizedStrides = UTILITY::ComputeStrideFromShape(fNormalizedShape);
      std::string normalizedIndex = "axis_" + std::to_string(fAxis) + " * " + std::to_string(normalizedStrides[0]);
      for (size_t i = fAxis + 1; i < fSize; i++) {
         normalizedIndex += " + axis_" + std::to_string(i) + " * "
            + std::to_string(normalizedStrides[i - fAxis]);
      }

      out << SP << "// Compute the mean\n";
      // Loop over the normalized dimensions
      for (size_t i = 0; i < fAxis; i++) {
         std::string iIdx = "axis_" + std::to_string(i);
         out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << inputShape;
         out << "[" << i << "]; " << iIdx << "++) {\n";
      }
         // Set sum = 0
         out << SP << SP << fType << " sum = 0.;\n";
         // loop over all the dims in [0, fAxis)
         for (size_t j = fAxis; j < fSize; j++) {
            std::string jIdx = "axis_" + std::to_string(j);
            out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape;
            out << "[" << j << "]; " << jIdx << "++) {\n";
         }
         out << SP << SP << SP << "sum += tensor_" << fNX << "[" << InputIndex << "];\n";
         for (size_t j = fAxis; j < fSize; j++) {
            out << SP << SP << "}\n";
         }
         out << SP << SP << "tensor_" << fNMean << "[" << axesIndex << "] = sum / " << fType << "(";
         out << fNormalizedLength << ");\n";
      for (size_t i = fAxis; i < fSize; i++) {
         out << SP << "}\n";
      }

      out << SP << "// Compute the inverse Standard Deviation\n";
      // Loop over the normalized dimensions
      for (size_t i = 0; i < fAxis; i++) {
         std::string iIdx = "axis_" + std::to_string(i);
         out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << inputShape;
         out << "[" << i << "]; " << iIdx << "++){\n";
      }
         // Set sum = 0
         out << SP << SP << fType << " sum = 0.;\n";
         // loop over all the dims in [0, fAxis)
         for (size_t j = fAxis; j < fSize; j++) {
            std::string jIdx = "axis_" + std::to_string(j);
            out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape;
            out << "[" << j << "]; " << jIdx << "++){\n";
         }
         out << SP << SP << SP << "sum += std::pow(tensor_" << fNX << "[" << InputIndex << "] - tensor_";
         out << fNMean << "[" << axesIndex << "], 2);\n";
         for (size_t j = fAxis; j < fSize; j++) {
            out << SP << SP << "}\n";
         }
         out << SP << SP << "tensor_" << fNInvStdDev << "[" << axesIndex << "] = 1 / std::sqrt(";
         out << "sum / " << fType << "(" << fNormalizedLength << ") + " << fAttrEpsilon << ");\n";
      for (size_t i = 0; i < fAxis; i++) {
         out << SP << "}\n";
      }

      out << SP << "// Y = Scale o NormalizedInput + B = Scale o InvStdDev (X - Mean) + B\n";
      // loop over all the dims in [0, fAxis)
      for (size_t i = 0; i < fAxis; i++) {
         std::string iIdx = "axis_" + std::to_string(i);
         out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << inputShape;
         out << "[" << i << "]; " << iIdx << "++){\n";
      }
         for (size_t j = fAxis; j < fSize; j++) {
            std::string jIdx = "axis_" + std::to_string(j);
            out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape;
            out << "[" << j << "]; " << jIdx << "++){\n";
         }
         out << SP << SP << SP << "tensor_" << fNY << "[" << InputIndex << "] = tensor_" << fNScale;
         out << "[" << normalizedIndex << "] * tensor_" << fNInvStdDev << "[" << axesIndex;
         out << "] * (tensor_" << fNX << "[" << InputIndex << "] - tensor_" << fNMean << "[";
         out << axesIndex << "]) + tensor_" << fNB << "[" << normalizedIndex << "];\n";
         for (size_t j = fAxis; j < fSize; j++) {
            out << SP << SP << "}\n";
         }
      for (size_t i = fAxis; i < fSize; i++) {
         out << SP << "}\n";
      }

      return out.str();
   }

   std::vector<std::string> GetStdLibs() override {
      return {"cmath"};
   }
};

ParserFuncSignature ParseLayerNormalisation = [](RModelParser_ONNX& parser, const onnx::NodeProto& nodeproto) -> std::unique_ptr<ROperator> {
   ETensorType input_type = ETensorType::UNDEFINED;
   const std::string input_name = nodeproto.input(0);
   if (parser.IsRegisteredTensorType(input_name)) {
      input_type = parser.GetTensorType(input_name);
   } else {
      throw
         std::runtime_error("TMVA::SOFIE ONNX Parser LayerNormalizaion op has input tensor "
            + input_name + " but its type is not yet registered");
   }

   int64_t axis = -1;
   float epsilon = 1e-5;
   int64_t stash_type = 1;
   for (size_t i = 0; i < nodeproto.attribute_size(); i++) {
      std::string attribute_name = nodeproto.attribute(i).name();
      if (attribute_name == "axis") {
         axis = nodeproto.attribute(i).i();
      } else if (attribute_name == "epsilon") {
         epsilon = nodeproto.attribute(i).f();
      } else if (attribute_name == "stash_type") {
         stash_type = nodeproto.attribute(i).i();
      }
   }
   size_t input_size = nodeproto.input_size();
   std::string name_scale = "";
   if (input_size > 1) {
      name_scale = nodeproto.input(1);
   }
   std::string name_bias = "";
   if (input_size > 2) {
      name_bias = nodeproto.input(2);
   }

   const std::string output_name = nodeproto.output(0);
   size_t output_size = nodeproto.output_size();
   std::string name_mean = "";
   if (output_size > 1) {
      name_mean = nodeproto.output(1);
   }
   std::string name_std = "";
   if (output_size > 2) {
      name_std = nodeproto.output(2);
   }

   std::unique_ptr<ROperator> op;
   switch(input_type) {
      case ETensorType::FLOAT:
         op.reset(new ROperator_LayerNormalization<float>(axis, epsilon, stash_type,
            input_name, name_scale, name_bias, output_name, name_mean, name_std));
         break;
      default:
         throw
            std::runtime_error("TMVA::SOFIE ONNX parser Operator with input type " + ConvertTypeToString(input_type) + " not supported.");
         break;
   }

   if (!parser.IsRegisteredTensorType(output_name)) {
      parser.RegisterTensorType(output_name, input_type);
   }

   return op;
};

void LayerNormalization() {
   RModelParser_ONNX parser;
   parser.RegisterOperator("LayerNormalization", ParseLayerNormalisation);

   RModel model = parser.Parse("./LayerNormalization3d.onnx");
   model.Generate();
   model.OutputGenerated();
}

