#include "./ConvTransposeBias2d.hxx"

#include <iostream>

const std::string path =
    "/home/ahmat/TMVAInference/Test/ConvTransposeBiasBatched";

int main() {
  {
     std::cout << "2d with bias\n";
     float* x = new float[18];
     for (size_t i = 0; i < 18; i++) x[i] = i;
     //TMVA_SOFIE_ConvTransposeBias2d::Session s(
        //path + "ConvTransposeBias2d.dat");
     std::vector<float> out = TMVA_SOFIE_ConvTransposeBias2d::infer(x);
     std::cout << "Size = " << out.size() << std::endl;
     for (size_t i = 0; i < out.size() ; i++) {
        std::cout << out[i] << " ";
     }
     std::cout << std::endl;
     delete[] x;
      /* 1.,   2.,   4.,   4.,   3.,   4.,   9.,  16.,  13.,   8.,  10.,  22.,
         37.,  28.,  16.,  10.,  21.,  34.,  25.,  14.,   7.,  14.,  22.,  16.,
          9.,   2.,   3.,   5.,   5.,   4.,   5.,  10.,  17.,  14.,   9.,  11.,
         23.,  38.,  29.,  17.,  11.,  22.,  35.,  26.,  15.,   8.,  15.,  23.,
         17.,  10.,  10.,  20.,  31.,  22.,  12.,  22.,  45.,  70.,  49.,  26.,
         37.,  76., 118.,  82.,  43.,  28.,  57.,  88.,  61.,  32.,  16.,  32.,
         49.,  34.,  18.,  11.,  21.,  32.,  23.,  13.,  23.,  46.,  71.,  50.,
         27.,  38.,  77., 119.,  83.,  44.,  29.,  58.,  89.,  62.,  33.,  17.,
         33.,  50.,  35.,  19.*/
  }

}
