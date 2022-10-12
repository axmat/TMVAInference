#include "LayerNormalization.hxx"

void TestLayerNormalization() {
   std::vector<float> x(12);
   for (size_t i = 0; i < 12; i++)
      x[i] = float(i);
   TMVA_SOFIE_LayerNormalization::Session s;
   auto y = s.infer(x.data());

   auto print = [](const std::vector<float>& v) {
      for (auto& val : v) {
         std::cout << val << " ";
      }
      std::cout << std::endl;
   };

   std::cout << "Input = ";
   print(x);
   std::cout << std::endl;

   std::cout << "LN(Input) = ";
   print(y);
   std::cout << std::endl;
}
