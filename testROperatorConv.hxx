#ifndef TEST_ROPERATOR_CONV
#define TEST_ROPERATOR_CONV

#include "TMVA/RTensor.hxx"
#include "testROperator.hxx"
#include "ROperatorConv.hxx"

#include <sstream>

template<typename T>
bool testROperatorConv_autopad_same(double tol);

template<typename T>
bool testROperatorConvWithPadding(double tol);

template<typename T>
bool testROperatorConvWithoutPadding(double tol);

template<typename T>
bool testROperatorConvStridesPadding(double tol);

template<typename T>
bool testROperatorConvStridesNoPadding(double tol);

template<typename T>
bool testROperatorConvStridesPaddingAlongOneDimension(double tol);

template<typename T>
bool testROperatorConvBatch(double tol);

template<typename T>
bool testROperatorConvGrouped(double tol);

template<typename T>
bool testROperatorConv(double tol) {
   bool failed = false;

   failed |= testROperatorConv_autopad_same<T>(tol);
   failed |= testROperatorConvWithPadding<T>(tol);
   failed |= testROperatorConvWithoutPadding<T>(tol);
   failed |= testROperatorConvStridesPadding<T>(tol);
   failed |= testROperatorConvStridesNoPadding<T>(tol);
   failed |= testROperatorConvStridesPaddingAlongOneDimension<T>(tol);
   failed |= testROperatorConvBatch<T>(tol);
   failed |= testROperatorConvGrouped<T>(tol);

   return failed;
}

template<typename T>
bool testROperatorConv_autopad_same(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorConv;

   RTensor<T> X({1, 1, 5, 5});
   std::iota(X.begin(), X.end(), 0.);

   RTensor<T> W({1, 1, 3, 3});
   std::fill(W.begin(), W.end(), 1.0);

   RTensor<T> B({});

   RTensor<T> Y({1, 1, 5, 5});
   T data[9] = {12.,  27., 24.,
                 63., 108., 81.,
                 72., 117., 84.,};
   RTensor<T> TrueY(data, {1, 1, 5, 5});

   ROperatorConv<T> conv("SAME_LOWER", {}, 0, {3, 3}, {}, {2, 2});
   conv.Forward_blas(X, W, B, Y);

   bool failed = !IsApprox(Y, TrueY, tol);
   std::stringstream ss;
   ss << "   ";
   ss << "Convolution with auto_pad =\"SAME_LOWER\" : Test ";
   ss << (failed? "Failed" : "Passed" );
   std::cout << ss.str() << std::endl;
   return failed;
}

template<typename T>
bool testROperatorConvWithPadding(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorConv;
   // Input
   RTensor<T> X({1, 1, 5, 5});
   std::iota(X.begin(), X.end(), 0.);
   // Kernel
   RTensor<T> W({1, 1, 3, 3});
   std::fill(W.begin(), W.end(), 1.0);
   // Bias
   RTensor<T> B({1});
   // Output
   RTensor<T> Y({1, 1, 5, 5});
   // True Output
   T data[25] = {12.,  21.,  27.,  33.,  24.,
                 33.,  54.,  63.,  72.,  51.,
                 63.,  99., 108., 117.,  81.,
                 93., 144., 153., 162., 111.,
                 72., 111., 117., 123.,  84.};
   RTensor<T> TrueY(data, {1, 1, 5, 5});

   ROperatorConv<T> conv("NOTSET",     // Autopad
                         {},           // dilations, default {}
                         0,            // group, default is 1
                         {3, 3},       // kernel shape
                         {1, 1, 1, 1}, // pads
                         {});          // strides, default {1, 1}
   conv.Forward_blas(X, W, B, Y);

   bool failed = !IsApprox(Y, TrueY, tol);
   std::stringstream ss;
   ss << "   ";
   ss << "Convolution with padding : Test ";
   ss << (failed? "Failed" : "Passed" );
   std::cout << ss.str() << std::endl;
   return failed;
}

template<typename T>
bool testROperatorConvWithoutPadding(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorConv;
   // Input
   RTensor<T> X({1, 1, 5, 5});
   std::iota(X.begin(), X.end(), 0.);
   // Kernel
   RTensor<T> W({1, 1, 3, 3});
   std::fill(W.begin(), W.end(), 1.0);
   // Bias
   RTensor<T> B({1});
   // Output
   RTensor<T> Y({1, 1, 3, 3});
   // True Output
   T data[9] = {54.,   63.,  72.,
                99.,  108., 117.,
                144., 153., 162.};
   RTensor<T> TrueY(data, {1, 1, 3, 3});

   ROperatorConv<T> conv("NOTSET",     // autopad
                         {},           // dilations, default {1, 1}
                         0,            // group, default 1
                         {3, 3},       // kernel shape
                         {0, 0, 0, 0}, // pads
                         {});          // strides, default {1, 1}
   conv.Forward_blas(X, W, B, Y);

   bool failed = !IsApprox(Y, TrueY, tol);
   std::stringstream ss;
   ss << "   ";
   ss << "Convolution without padding : Test ";
   ss << (failed? "Failed" : "Passed" );
   std::cout << ss.str() << std::endl;
   return failed;
}

template<typename T>
bool testROperatorConvStridesPadding(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorConv;
   // Input
   RTensor<T> X({1, 1, 7, 5});
   std::iota(X.begin(), X.end(), 0.);
   // Kernel
   RTensor<T> W({1, 1, 3, 3});
   std::fill(W.begin(), W.end(), 1.0);
   // Bias
   RTensor<T> B({1});
   // Output
   RTensor<T> Y({1, 1, 4, 3});
   // True Output
   T data[12] = {12.,   27.,  24.,
                 63.,  108.,  81.,
                 123., 198., 141.,
                 112., 177., 124.};
   RTensor<T> TrueY(data, {1, 1, 4, 3});

   ROperatorConv<T> conv("NOTSET",     // autopad
                         {},           // dilations, default {1, 1}
                         0,            // group, default 1
                         {3, 3},       // kernel shape
                         {1, 1, 1, 1}, // pads
                         {2, 2});      // strides
   conv.Forward_blas(X, W, B, Y);

   bool failed = !IsApprox(Y, TrueY, tol);
   std::stringstream ss;
   ss << "   ";
   ss << "Convolution with strides=2 and padding : Test ";
   ss << (failed? "Failed" : "Passed" );
   std::cout << ss.str() << std::endl;
   return failed;
}

template<typename T>
bool testROperatorConvStridesNoPadding(double tol){
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorConv;
   // Input
   RTensor<T> X({1, 1, 7, 5});
   std::iota(X.begin(), X.end(), 0.);
   // Kernel
   RTensor<T> W({1, 1, 3, 3});
   std::fill(W.begin(), W.end(), 1.0);
   // Bias
   RTensor<T> B({1});
   // Output
   RTensor<T> Y({1, 1, 3, 2});
   // True Output
   T data[6] = {54.,   72.,
                144., 162.,
                234., 252.};
   RTensor<T> TrueY(data, {1, 1, 3, 2});

   ROperatorConv<T> conv("NOTSET",     // autopad
                         {},           // dilations, default {1, 1}
                         0,            // group, default 1
                         {3, 3},       // kernel shape
                         {0, 0, 0, 0}, // pads
                         {2, 2});      // strides
   conv.Forward_blas(X, W, B, Y);

   bool failed = !IsApprox(Y, TrueY, tol);
   std::stringstream ss;
   ss << "   ";
   ss << "Convolution with strides=2 and no padding : Test ";
   ss << (failed? "Failed" : "Passed" );
   std::cout << ss.str() << std::endl;
   return failed;
};

template<typename T>
bool testROperatorConvStridesPaddingAlongOneDimension(double tol){
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorConv;
   // Input
   RTensor<T> X({1, 1, 7, 5});
   std::iota(X.begin(), X.end(), 0.);
   // Kernel
   RTensor<T> W({1, 1, 3, 3});
   std::fill(W.begin(), W.end(), 1.0);
   // Bias
   RTensor<T> B({1});
   // Output
   RTensor<T> Y({1, 1, 4, 2});
   // True Output
   T data[8] = {21.,   33.,
                99.,  117.,
                189., 207.,
                171., 183.};
   RTensor<T> TrueY(data, {1, 1, 4, 2});

   ROperatorConv<T> conv("NOTSET",     // autopad
                         {},           // dilations, default {1, 1}
                         0,            // group, default 1
                         {3, 3},       // kernel shape
                         {1, 0, 1, 0}, // pads
                         {2, 2});      // strides
   conv.Forward_blas(X, W, B, Y);

   bool failed = !IsApprox(Y, TrueY, tol);
   std::stringstream ss;
   ss << "   ";
   ss << "Convolution with strides=2 and padding along only one dimension : Test ";
   ss << (failed? "Failed" : "Passed" );
   std::cout << ss.str() << std::endl;
   return failed;
};

template<typename T>
bool testROperatorConvBatch(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorConv;
   // Input
   RTensor<T> X({2, 1, 5, 5});
   for(std::size_t n=0; n < 2; n++) {
      T val = 0.0;
      for(std::size_t h=0; h < 5; h++) {
         for(std::size_t w=0; w < 5; w++) {
            X(n, 0, h, w) = val;
            val++;
         }
      }
   }
   // Kernel
   RTensor<T> W({1, 1, 3, 3});
   std::fill(W.begin(), W.end(), 1.0);
   // Bias
   RTensor<T> B({1});
   // Output
   RTensor<T> Y({1, 2, 5, 5});
   // True Output
   T data[50] = {12.,  21.,  27.,  33.,  24.,
                 33.,  54.,  63.,  72.,  51.,
                 63.,  99., 108., 117.,  81.,
                 93., 144., 153., 162., 111.,
                 72., 111., 117., 123.,  84.,
                 12.,  21.,  27.,  33.,  24.,
                 33.,  54.,  63.,  72.,  51.,
                 63.,  99., 108., 117.,  81.,
                 93., 144., 153., 162., 111.,
                 72., 111., 117., 123.,  84.};
   RTensor<T> TrueY(data, {1, 2, 5, 5});

   ROperatorConv<T> conv("NOTSET",     // autopad
                         {},           // dilations
                         0,            // group, default 1
                         {3, 3},       // kernel shape
                         {1, 1, 1, 1}, // pads
                         {});          // strides, default {1, 1}
   conv.Forward_blas(X, W, B, Y);

   bool failed = !IsApprox(Y, TrueY, tol);
   std::stringstream ss;
   ss << "   ";
   ss << "Convolution with padding and batched input: Test ";
   ss << (failed? "Failed" : "Passed" );
   std::cout << ss.str() << std::endl;
   return failed;
}

template<typename T>
bool testROperatorConvGrouped(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorConv;
   // Input
   RTensor<T> X({1, 4, 2, 2});
   std::iota(X.begin(), X.end(), 0.0);
   // Kernel
   RTensor<T> W({4, 2, 2, 2});
   std::iota(W.begin(), W.end(), 0.0);
   // Bias
   RTensor<T> B({4});
   std::iota(B.begin(), B.end(), 0.0);
   // Output
   RTensor<T> Y({4, 1, 1, 1});
   // True Output
   T data[4] = {140., 365., 1838., 2575.};
   RTensor<T> TrueY(data, {4, 1, 1, 1});

   ROperatorConv<T> conv("NOTSET",     // autopad
                         {},           // dilations
                         2,            // group
                         {2, 2},       // kernel shape
                         {0, 0, 0, 0}, // pads
                         {});          // strides, default {1, 1}
   conv.Forward_blas(X, W, B, Y);

   bool failed = !IsApprox(Y, TrueY, tol);
   std::stringstream ss;
   ss << "   ";
   ss << "Grouped Convolution: Test ";
   ss << (failed? "Failed" : "Passed" );
   std::cout << ss.str() << std::endl;
   return failed;
}

#endif
