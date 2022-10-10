#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// OP types
// enum class OpType{kConv, kGemm, kRelu};

// Opertaor
struct Op {
  virtual const std::string &Name() = 0;
  virtual const std::string &Meta() = 0;
  virtual ~Op() {}
};

struct Conv : Op {
  std::string name = "Conv";
  std::string meta;
  const std::string &Name() override { return name; }
  const std::string &Meta() override { return meta; }
  Conv(const std::string &data) : meta(data) {}
};

struct Relu : Op {
  std::string name = "Relu";
  std::string meta;
  const std::string &Name() override { return name; }
  const std::string &Meta() override { return meta; }
  Relu(const std::string &data) : meta(data) {}
};

// Onnx node
struct OnnxNode {
  std::string type;
  std::string meta;
  OnnxNode(const std::string &nodetype, const std::string &data)
      : type(nodetype), meta(data) {}
};

auto parseConv = [](const OnnxNode &node) -> std::unique_ptr<Op> {
  if (node.type != "Conv") {
    throw std::runtime_error("Node must be of type Conv");
  }
  std::cout << node.type << node.meta << std::endl;
  std::unique_ptr<Op> op;
  op.reset(new Conv(node.meta));
  return op;
};

auto parseRelu = [](const OnnxNode &node) -> std::unique_ptr<Op> {
  if (node.type != "Relu") {
    throw std::runtime_error("Node must be of type Relu");
  }
  std::cout << node.type << node.meta << std::endl;
  std::unique_ptr<Op> op;
  op.reset(new Relu(node.meta));
  std::cout << op.get()->Name() << op.get()->Meta() << std::endl;
  return op;
};

struct Model {
  std::vector<std::unique_ptr<Op>> ops;

  void AddOp(std::unique_ptr<Op> op) { ops.push_back(std::move(op)); }

  void Print() {
    std::cout << "Model:\n";
    for (auto &op : ops) {
      std::cout << "[" << op.get()->Name() << " " << op.get()->Meta() << "]"
                << std::endl;
    }
  }
};


class NotRegisteredError : std::exception {
private:
   std::string msg;

public:
   NotRegisteredError(const std::string& s) : msg{"Error - Operator " + s + " is not registered."} {}

   const char* what() const noexcept override { return msg.c_str();}
};

struct Parser {
  using FuncSignature = std::function<std::unique_ptr<Op>(const OnnxNode &)>;
  // registered ops
  std::unordered_map<std::string, FuncSignature> registeredOps;

  // Register an operators
  void RegisterOp(const std::string &name, FuncSignature func) {
    registeredOps[name] = func;
  }

  // Is registered
  bool IsRegisteredOp(const std::string &name) {
    return registeredOps.find(name) != registeredOps.end();
  }

  // Registered ops
  std::vector<std::string> RegisteredOps() {
    std::vector<std::string> ops;
    ops.reserve(registeredOps.size());
    for (auto &it : registeredOps) {
      ops.emplace_back(it.first);
    }
    return ops;
  }

  // Parse Op
  std::unique_ptr<Op> ParseOp(const OnnxNode &node) {
    // std::cout << node.type << node.meta << std::endl;
    std::string type = node.type;
    auto it = registeredOps.find(type);
    if (it == registeredOps.end()) {
      throw NotRegisteredError(type);
      //throw std::runtime_error("ParseNode: operator " + type +
      //                         " not registered.");
    }
    return it->second(node);
    // registeredOps[type];
  }

  Parser() {
    // Implemented ops
    RegisterOp("Conv", parseConv);
    RegisterOp("Relu", parseRelu);
  }

  Model Parse(const std::string &filename = "test.model") {
    std::ifstream file(filename);
    std::string type, meta;
    size_t n;
    file >> n;
    Model model;
    while (n--) {
      file >> type;
      file >> meta;
      // std::cout << type << meta << std::endl;
      OnnxNode node(type, meta);
      model.AddOp(ParseOp(node));
    }
    return model;
  }
};

int main() {
  Parser parser;

  std::cout << "IsRegisteredOp:\n";
  std::cout << "Conv: " << std::boolalpha << parser.IsRegisteredOp("Conv")
            << std::endl;
  std::cout << "Gemm: " << std::boolalpha << parser.IsRegisteredOp("Gemm")
            << std::endl;

  std::cout << "List of registered operators\n";
  for (auto &name : parser.RegisteredOps()) {
    std::cout << name << std::endl;
  }
  std::cout << std::endl;

  Model model = parser.Parse();
  model.Print();

  // test unregistered Gemm
  try {
    Model model = parser.Parse("test_gemm.model");
  } catch (NotRegisteredError& e) {
    std::cout << e.what() << std::endl;
  }
  std::cout << std::endl;

  // registering a new op
  // Define a new op
  struct Gemm : Op {
    std::string name = "Gemm";
    std::string meta;

    const std::string &Name() override { return name; }
    const std::string &Meta() override { return meta; }
    Gemm(const std::string &data) : meta(data) {}
  };
  // Define a function to parse the node
  auto parseGemm = [](const OnnxNode &node) -> std::unique_ptr<Op> {
    if (node.type != "Gemm") {
      throw std::runtime_error("Node must be of type Gemm");
    }
    std::cout << node.type << node.meta << std::endl;
    std::unique_ptr<Op> op;
    op.reset(new Gemm(node.meta));
    return op;
  };
  // Register it
  parser.RegisterOp("Gemm", parseGemm);
  std::cout << "List of registered operators\n";
  for (auto &name : parser.RegisteredOps()) {
    std::cout << name << std::endl;
  }
  std::cout << std::endl;

  // Parse the model
  // Now the model can parse a node of type Gemm
  { Model model = parser.Parse("test_gemm.model"); }

}
