#include "./Sqrt.hxx"
#include "./Reciprocal.hxx"

#include <iostream>

const std::string path =
    "/home/ahmat/TMVAInference/Test/UnaryOperators/";

int main() {
  {
    std::cout << "Sqrt\n";
   float x[] = {
0.8344, 0.4716, 0.6226, 0.8448, 0.2483, 0.9467};

    TMVA_SOFIE_Sqrt::Session s(path + "Sqrt.dat");
    std::vector<float> out = s.infer(x);
    std::cout << "Size = " << out.size() << std::endl;
    for (size_t i = 0; i < out.size(); i++) {
      std::cout << out[i] << " ";
    }
    std::cout << std::endl;
    // 0.9135, 0.6868, 0.7891, 0.9191, 0.4983, 0.9730
  }
  {
    std::cout << "Reciprocal\n";
   float x[] = {
1.2691, -1.2160,  0.6393, -0.4438,  0.8065,  0.2011};

    TMVA_SOFIE_Reciprocal::Session s(path + "Reciprocal.dat");
    std::vector<float> out = s.infer(x);
    std::cout << "Size = " << out.size() << std::endl;
    for (size_t i = 0; i < out.size(); i++) {
      std::cout << out[i] << " ";
    }
    std::cout << std::endl;
   // 0.7879, -0.8223,  1.5643, -2.2532,  1.2399,  4.9723
  }

}
