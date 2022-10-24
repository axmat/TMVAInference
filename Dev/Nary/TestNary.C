#include "./MaxMultidirectionalBroadcast.hxx"

#include <numeric>

void TestMax() {
  auto print = [](const std::vector<float> &v) {
    for (auto &val : v) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
  };


  {
   std::cout << "Max" << std::endl;
    std::vector<float> a({0.35974154, -2.20873388,  0.95746274});
    std::vector<float> b({0.75901985, -0.46544461, -0.34920575, -0.1460754 ,  0.08269051, -0.70045695});
    std::vector<float> c({-0.41468981, -0.46591926,  0.56172534,  0.05616931});
    TMVA_SOFIE_MaxMultidirectionalBroadcast::Session s;
   // infer(float* b, float* c, float* a)
    auto y = s.infer(b.data(), c.data(), a.data());

    std::cout << "a = ";
    print(a);
    std::cout << std::endl;

    std::cout << "b = ";
    print(b);
    std::cout << std::endl;

    std::cout << "c = ";
    print(c);
    std::cout << std::endl;

    std::cout << "Y = ";
    print(y);
    std::cout << std::endl;

    size_t size = y.size();
    std::vector<float> true_y({0.7590198503375636, 0.7590198503375636, 0.7590198503375636, 0.7590198503375636,
      -0.41468980634400543, -0.465444611539521, 0.5617253354820355, 0.05616930535561424, 0.9574627354854222,
      0.9574627354854222, 0.9574627354854222, 0.9574627354854222, 0.3597415448611981, 0.3597415448611981,
      0.5617253354820355, 0.3597415448611981, 0.08269051091686609, 0.08269051091686609, 0.5617253354820355,
      0.08269051091686609, 0.9574627354854222, 0.9574627354854222, 0.9574627354854222, 0.9574627354854222});
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
    std::vector<float> a({0.35974154, -2.20873388,  0.95746274});
    std::vector<float> b({0.75901985, -0.46544461, -0.34920575, -0.1460754 ,  0.08269051, -0.70045695});
    std::vector<float> c({-0.41468981, -0.46591926,  0.56172534,  0.05616931});
    TMVA_SOFIE_MaxMultidirectionalBroadcast::Session s;
   // infer(float* b, float* c, float* a)
    auto y = s.infer(b.data(), c.data(), a.data());

    std::cout << "a = ";
    print(a);
    std::cout << std::endl;

    std::cout << "b = ";
    print(b);
    std::cout << std::endl;

    std::cout << "c = ";
    print(c);
    std::cout << std::endl;

    std::cout << "Y = ";
    print(y);
    std::cout << std::endl;

    size_t size = y.size();
    std::vector<float> true_y({0.7590198503375636, 0.7590198503375636, 0.7590198503375636, 0.7590198503375636,
      -0.41468980634400543, -0.465444611539521, 0.5617253354820355, 0.05616930535561424, 0.9574627354854222,
      0.9574627354854222, 0.9574627354854222, 0.9574627354854222, 0.3597415448611981, 0.3597415448611981,
      0.5617253354820355, 0.3597415448611981, 0.08269051091686609, 0.08269051091686609, 0.5617253354820355,
      0.08269051091686609, 0.9574627354854222, 0.9574627354854222, 0.9574627354854222, 0.9574627354854222});
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
