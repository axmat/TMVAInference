#include "./RangeFloat.hxx"
#include "./RangeInt.hxx"

#include <numeric>

void TestRange()
{
   {
      std::cout << "RangeFloat" << std::endl;
      std::vector<float> start({1});
      std::vector<float> limit({10});
      std::vector<float> delta({2});
      TMVA_SOFIE_RangeFloat::Session s;
      auto y = s.infer(start.data(), limit.data(), delta.data());
      size_t size = y.size();
      std::cout << "output y = " << std::endl;
      for (size_t i = 0; i < size; i++)
         std::cout << y[i] << " ";
      std::cout << std::endl;

      std::vector<float> true_y({1, 3, 5, 7, 9});
      for (size_t i = 0; i < size; i++) {
         if (std::abs(y[i] - true_y[i]) > 1e-3) {
            std::cout << "Diff output\n";
            break;
         }
         if (i + 1 == size) {
            std::cout << "Same output\n";
         }
      }
   }

   {
      std::cout << "RangeInt" << std::endl;
      std::vector<int64_t> start({1});
      std::vector<int64_t> limit({10});
      std::vector<int64_t> delta({2});
      TMVA_SOFIE_RangeInt::Session s;
      auto y = s.infer(start.data(), limit.data(), delta.data());
      size_t size = y.size();
      std::cout << "output y = " << std::endl;
      for (size_t i = 0; i < size; i++)
         std::cout << y[i] << " ";
      std::cout << std::endl;

      std::vector<int64_t> true_y({1, 3, 5, 7, 9});
      for (size_t i = 0; i < size; i++) {
         if (std::abs(y[i] - true_y[i]) > 1e-3) {
            std::cout << "Diff output\n";
            break;
         }
         if (i + 1 == size) {
            std::cout << "Same output\n";
         }
      }
   }
}
