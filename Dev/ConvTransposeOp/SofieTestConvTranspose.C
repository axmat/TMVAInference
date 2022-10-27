#include "./ConvTransposeM.hxx"
#include "./Output.hxx"

#include <numeric>
#include <chrono>

void SofieTestConvTranspose() {
   size_t N = 4 * 3 * 30 * 30;
   std::vector<float> x(N);
	std::fill(x.begin(), x.end(), 1.);
   TMVA_SOFIE_ConvTransposeM::Session s;

	std::cout << "Runing inference" << std::endl;
   auto t0 = std::chrono::high_resolution_clock::now();
   auto y = s.infer(x.data());
   auto t1 = std::chrono::high_resolution_clock::now();

   size_t size = output.size();
   if (size != y.size()) {
      std::cout << "Different output size" << std::endl;
   } else {
      std::cout << "Same output size " << size << std::endl;
      for (size_t i = 0; i < size; i++) {
         if (y[i] != output[i]) {
            std::cout << "Different output at index " << i << std::endl;
            std::cout << y[i] << " and " << output[i] << std::endl;
            break;
         } else if (i + 1 == size) {
            std::cout << "Same output" << std::endl;
         }
      }
   }

   std::cout << "Time (s): " << std::chrono::duration<double>(t1 - t0).count() << std::endl;
}
