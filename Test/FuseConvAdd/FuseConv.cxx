#include "./FuseConvTransposeAdd.hxx"

#include <iostream>

const std::string path =
    "/home/ahmat/TMVAInference/Test/FuseConvAdd/";

int main() {
  {
    std::cout << "Fuse ConvTranspose Add\n";
   float x[15];
   for (size_t i = 0; i < 15; i++)
      x[i] = i;

    TMVA_SOFIE_Conv2DTranspose_Relu_Sigmoid::Session s(path + "FuseConvTransposeAdd.dat");
    std::vector<float> out = s.infer(x);
    std::cout << "Size = " << out.size() << std::endl;
    for (size_t i = 0; i < out.size(); i++) {
      std::cout << out[i] << " ";
    }
    std::cout << std::endl;
  }
  }
