#include "Sign.hxx"

void Run() {
   std::vector<float> x({0.1, 0., -3, 2.5, 10.9});
   TMVA_SOFIE_Sign::Session s;
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

   std::cout << "Sign(Input) = ";
   print(y);
   std::cout << std::endl;
}
