#include "./MaxMultidirectionalBroadcast.hxx"
#include "./MeanMultidirectionalBroadcast.hxx"
#include "./MinMultidirectionalBroadcast.hxx"
#include "./SumMultidirectionalBroadcast.hxx"

#include <numeric>

void TestNary()
{
   {
      std::cout << "Max" << std::endl;
      std::vector<float> a({0.35974154, -2.20873388, 0.95746274});
      std::vector<float> b({0.75901985, -0.46544461, -0.34920575, -0.1460754, 0.08269051, -0.70045695});
      std::vector<float> c({-0.41468981, -0.46591926, 0.56172534, 0.05616931});
      TMVA_SOFIE_MaxMultidirectionalBroadcast::Session s;
      auto y = s.infer(b.data(), c.data(), a.data());

      size_t size = y.size();
      std::vector<float> true_y({0.7590198503375636,   0.7590198503375636,  0.7590198503375636, 0.7590198503375636,
                                 -0.41468980634400543, -0.465444611539521,  0.5617253354820355, 0.05616930535561424,
                                 0.9574627354854222,   0.9574627354854222,  0.9574627354854222, 0.9574627354854222,
                                 0.3597415448611981,   0.3597415448611981,  0.5617253354820355, 0.3597415448611981,
                                 0.08269051091686609,  0.08269051091686609, 0.5617253354820355, 0.08269051091686609,
                                 0.9574627354854222,   0.9574627354854222,  0.9574627354854222, 0.9574627354854222});
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
      std::cout << "Min" << std::endl;
      std::vector<float> a({0.35974154, -2.20873388, 0.95746274});
      std::vector<float> b({0.75901985, -0.46544461, -0.34920575, -0.1460754, 0.08269051, -0.70045695});
      std::vector<float> c({-0.41468981, -0.46591926, 0.56172534, 0.05616931});
      TMVA_SOFIE_MinMultidirectionalBroadcast::Session s;
      auto y = s.infer(b.data(), c.data(), a.data());

      size_t size = y.size();
      std::vector<float> true_y({
-0.41468981, -0.46591926, 0.35974154, 0.05616931, -2.20873388, -2.20873388, -2.20873388, -2.20873388, -0.41468981, -0.46591926, -0.34920575, -0.34920575, -0.41468981, -0.46591926, -0.1460754, -0.1460754, -2.20873388, -2.20873388, -2.20873388, -2.20873388, -0.70045695, -0.70045695, -0.70045695, -0.70045695});

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
      std::cout << "Mean" << std::endl;
      std::vector<float> a({0.35974154, -2.20873388, 0.95746274});
      std::vector<float> b({0.75901985, -0.46544461, -0.34920575, -0.1460754, 0.08269051, -0.70045695});
      std::vector<float> c({-0.41468981, -0.46591926, 0.56172534, 0.05616931});
      TMVA_SOFIE_MeanMultidirectionalBroadcast::Session s;
      auto y = s.infer(b.data(), c.data(), a.data());

      size_t size = y.size();
      std::vector<float> true_y({0.23469052666666665, 0.21761404333333334, 0.5601622433333333, 0.39164356666666666, -1.0296227666666666, -1.04669925, -0.7041510500000001, -0.8726697266666666, 0.06452239333333333, 0.04744591, 0.38999410999999995, 0.22147543333333333, -0.06700788999999999, -0.08408437333333334, 0.2584638266666667, 0.08994515000000002, -0.84691106, -0.8639875433333334, -0.5214393433333334, -0.6899580200000001, -0.05256133999999999, -0.06963782333333333, 0.2729103766666667, 0.10439170000000002});

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
      std::cout << "Sum" << std::endl;
      std::vector<float> a({0.35974154, -2.20873388, 0.95746274});
      std::vector<float> b({0.75901985, -0.46544461, -0.34920575, -0.1460754, 0.08269051, -0.70045695});
      std::vector<float> c({-0.41468981, -0.46591926, 0.56172534, 0.05616931});
      TMVA_SOFIE_SumMultidirectionalBroadcast::Session s;
      auto y = s.infer(b.data(), c.data(), a.data());

      size_t size = y.size();
      std::vector<float> true_y({0.7040715799999999, 0.65284213, 1.68048673, 1.1749307, -3.0888683, -3.1400977500000002, -2.1124531500000003, -2.61800918, 0.19356718, 0.14233773, 1.1699823299999998, 0.6644263, -0.20102366999999996, -0.25225312, 0.77539148, 0.26983545000000003, -2.54073318, -2.5919626300000003, -1.5643180300000001, -2.06987406, -0.15768401999999998, -0.20891347, 0.81873113, 0.31317510000000004});

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
