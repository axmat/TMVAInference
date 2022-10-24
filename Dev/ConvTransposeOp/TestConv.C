#include "./Conv2DTranspose_Relu_Sigmoid.hxx"
#include <numeric>

auto print = [](const std::vector<float>& v) {
	for (auto& val : v) {
		std::cout << val << " ";
	}
	std::cout << std::endl;
};

void TestConv() {
   std::vector<float> x(15);
	std::iota(x.begin(), x.end(), 0.);
   TMVA_SOFIE_Conv2DTranspose_Relu_Sigmoid::Session s;

   std::cout << "Input = ";
   print(x);
   std::cout << std::endl;

	std::cout << "Runing inference" << std::endl;
   auto y = s.infer(x.data());

   std::cout << "Output = ";
   print(y);
   std::cout << std::endl;
}
