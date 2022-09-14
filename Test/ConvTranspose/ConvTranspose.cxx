#include "./ConvTranspose1d.hxx"
#include "./ConvTranspose2d.hxx"
#include "./ConvTranspose3d.hxx"
#include "./ConvTransposeBias2d.hxx"
#include "./ConvTransposeGrouped2d.hxx"

#include <iostream>

const std::string path =
    "/Users/ahmat/cern/tmva_inference/SofieTest/ConvTranspose";

int main() {
  {
    std::cout << "Grouped 2d\n";
    float *x = new float[18];
    for (size_t i = 0; i < 18; i++)
      x[i] = i;
    TMVA_SOFIE_ConvTransposeGrouped2d::Session s(path +
                                                 "ConvTransposeGrouped2d.dat");
    std::vector<float> out = s.infer(x);
    std::cout << "Size = " << out.size() << std::endl;
    for (size_t i = 0; i < out.size(); i++) {
      std::cout << out[i] << " ";
    }
    std::cout << std::endl;
    delete[] x;
    // 0 1 3 3 2 3 8 15 12 7 9 21 36 27 15 9 20 33 24 13 6 13 21 15 8 0 1 3 3 2
    // 3 8 15 12 7 9 21 36 27 15 9 20 33 24 13 6 13 21 15 8 9 19 30 21 11 21 44
    // 69 48 25 36 75 117 81 42 27 56 87 60 31 15 31 48 33 17 9 19 30 21 11 21
    // 44 69 48 25 36 75 117 81 42 27 56 87 60 31 15 31 48 33 17
  }

  /*
  {
     std::cout << "2d with bias\n";
     float* x = new float[9];
     for (size_t i = 0; i < 9; i++) x[i] = i;
     TMVA_SOFIE_ConvTransposeBias2d::Session s(
        "/Users/ahmat/cern/sofie-test/test/ConvTransposeBias2d.dat");
     std::vector<float> out = s.infer(x);
     std::cout << "Size = " << out.size() << std::endl;
     for (size_t i = 0; i < out.size() ; i++) {
        std::cout << out[i] << " ";
     }
     std::cout << std::endl;
     delete[] x;
     // 1 2 4 4 3 4 9 16 13 8 10 22 37 28 16 10 21 34 25 14 7 14 22 16 9 2 3 5 5
  4 5 10 17 14 9 11 23 38 29 17 11 22 35 26 15 8 15 23 17 10
  }*/

  {
    std::cout << "1d\n";
    float *x = new float[3];
    for (size_t i = 0; i < 3; i++)
      x[i] = i;
    TMVA_SOFIE_ConvTranspose1d::Session s(path + "ConvTranspose1d.dat");
    std::vector<float> out = s.infer(x);
    std::cout << "Size = " << out.size() << std::endl;
    for (size_t i = 0; i < out.size(); i++) {
      std::cout << out[i] << " ";
    }
    std::cout << std::endl;
    delete[] x;
  }

  {
    std::cout << "2d\n";
    float *x = new float[3];
    for (size_t i = 0; i < 3; i++)
      x[i] = i;
    TMVA_SOFIE_ConvTranspose2d::Session s(path + "ConvTranspose2d.dat");
    std::vector<float> out = s.infer(x);
    std::cout << "Size = " << out.size() << std::endl;
    for (size_t i = 0; i < out.size(); i++) {
      std::cout << out[i] << " ";
    }
    std::cout << std::endl;
    delete[] x;
  }

  {
    std::cout << "3d\n";
    float *x = new float[60];
    for (size_t i = 0; i < 9; i++)
      x[i] = i;

    TMVA_SOFIE_ConvTranspose2d::Session s(path + "ConvTranspose2d.dat");
    std::vector<float> out = s.infer(x);
    std::cout << "Size = " << out.size() << std::endl;
    for (size_t i = 0; i < out.size(); i++) {
      std::cout << out[i] << " ";
    }
    std::cout << std::endl;

    delete[] x;
  }
}
