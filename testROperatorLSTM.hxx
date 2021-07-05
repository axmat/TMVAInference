#ifndef TEST_ROPERATOR_LSTM
#define TEST_ROPERATOR_LSTM

#include <iostream>
#include <sstream>
#include <algorithm>

#include "testROperator.hxx"
#include "ROperatorLSTM.hxx"
#include <TMVA/RTensor.hxx>

template<typename T>
bool testROperatorLSTM_defaults(double tol);

template<typename T>
bool testROperatorLSTM_initial_bias(double tol);

template<typename T>
bool testROperatorLSTM_peepholes(double tol);

template<typename T>
bool testROperatorLSTM_batchwise(double tol);

template<typename T>
bool testROperatorLSTM_bidirectional(double tol);

template<typename T>
bool testROperatorLSTM(double tol) {
   bool failed = false;

   failed |= testROperatorLSTM_defaults<T>(tol);
   failed |= testROperatorLSTM_initial_bias<T>(tol);
   failed |= testROperatorLSTM_peepholes<T>(tol);
   failed |= testROperatorLSTM_batchwise<T>(tol);
   failed |= testROperatorLSTM_bidirectional<T>(tol);

   return failed;
}

template<typename T>
bool testROperatorLSTM_defaults(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorLSTM;
   const size_t seq_length = 3;
   const size_t batch_size = 1;
   const size_t input_size = 2;
   const size_t hidden_size = 3;
   const size_t num_directions = 1;

   RTensor<T> x({seq_length, batch_size, input_size});
   std::iota(x.begin(), x.end(), 1.);

   RTensor<T> w({num_directions, 4*hidden_size, input_size});
   std::fill(w.begin(), w.end(), 0.1);

   RTensor<T> b({});

   RTensor<T> r({num_directions, 4*hidden_size, hidden_size});
   std::fill(r.begin(), r.end(), 0.1);

   RTensor<size_t> sequance_lens({});

   RTensor<T> initial_h({});

   RTensor<T> initial_c({});

   RTensor<T> p({});

   RTensor<T> y({seq_length, num_directions, batch_size, hidden_size});
   RTensor<T> y_h({num_directions, batch_size, hidden_size});
   RTensor<T> y_c({});

   T true_y_data[seq_length * num_directions * batch_size * hidden_size] = {
      0.09524119, 0.09524119, 0.09524119,
      0.32869044, 0.32869044, 0.32869044,
      0.60042989, 0.60042989, 0.60042989};
   RTensor<T> true_y(true_y_data, {seq_length, num_directions, batch_size, hidden_size});

   T true_y_h_data[num_directions * batch_size * hidden_size] = {
      0.60042989, 0.60042989, 0.60042989};
   RTensor<T> true_y_h(true_y_h_data, {num_directions, batch_size, hidden_size});

   ROperatorLSTM<T> lstm({{}, {}, {}, 0.0, "forward", hidden_size, 0, 0});
   lstm.Forward_blas(x, w, r, b, sequance_lens, initial_h, initial_c, p, y, y_h, y_c);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol);
   std::cout << "   LSTM defaults : Test ";
   std::cout << (failed? "Failed" : "Passed") << std::endl;
   return failed;
}

template<typename T>
bool testROperatorLSTM_initial_bias(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorLSTM;
   const size_t seq_length = 3;
   const size_t batch_size = 1;
   const size_t input_size = 3;
   const size_t hidden_size = 4;
   const size_t num_directions = 1;

   RTensor<T> x({seq_length, batch_size, input_size});
   std::iota(x.begin(), x.end(), 1.);

   RTensor<T> w({num_directions, 4*hidden_size, input_size});
   std::fill(w.begin(), w.end(), 0.1);

   RTensor<T> b({num_directions, 8 * hidden_size});
   std::fill(b.begin(), b.begin() + 4*hidden_size, 0.1);
   std::fill(b.begin() + 4*hidden_size, b.end(), 0.);

   RTensor<T> r({num_directions, 4*hidden_size, hidden_size});
   std::fill(r.begin(), r.end(), 0.1);

   RTensor<size_t> sequance_lens({});

   RTensor<T> initial_h({});

   RTensor<T> initial_c({});

   RTensor<T> p({});

   RTensor<T> y({seq_length, num_directions, batch_size, hidden_size});
   RTensor<T> y_h({num_directions, batch_size, hidden_size});
   RTensor<T> y_c({});

   T true_y_data[seq_length * num_directions * batch_size * hidden_size] = {
      0.25606444, 0.25606444, 0.25606444, 0.25606444,
      0.68688357, 0.68688357, 0.68688357, 0.68688357,
      0.90747154, 0.90747154, 0.90747154, 0.90747154};
   RTensor<T> true_y(true_y_data, {seq_length, num_directions, batch_size, hidden_size});

   T true_y_h_data[num_directions * batch_size * hidden_size] = {
      0.90747154, 0.90747154, 0.90747154, 0.90747154};
   RTensor<T> true_y_h(true_y_h_data, {num_directions, batch_size, hidden_size});

   ROperatorLSTM<T> lstm({{}, {}, {}, 0.0, "forward", hidden_size, 0, 0});
   lstm.Forward_blas(x, w, r, b, sequance_lens, initial_h, initial_c, p, y, y_h, y_c);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol);
   std::cout << "   LSTM initial bias : Test ";
   std::cout << (failed? "Failed" : "Passed") << std::endl;
   return failed;
}

template<typename T>
bool testROperatorLSTM_peepholes(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorLSTM;
   const size_t seq_length = 1;
   const size_t batch_size = 2;
   const size_t input_size = 4;
   const size_t hidden_size = 3;
   const size_t num_directions = 1;

   RTensor<T> x({seq_length, batch_size, input_size});
   std::iota(x.begin(), x.end(), 1.);

   RTensor<T> w({num_directions, 4*hidden_size, input_size});
   std::fill(w.begin(), w.end(), 0.1);

   RTensor<T> b({num_directions, 8 * hidden_size});
   std::fill(b.begin(), b.end(), 0.);

   RTensor<T> r({num_directions, 4*hidden_size, hidden_size});
   std::fill(r.begin(), r.end(), 0.1);

   RTensor<size_t> sequance_lens({2});
   sequance_lens(0) = 1;
   sequance_lens(1) = 1;

   RTensor<T> initial_h({num_directions, batch_size, hidden_size});
   std::fill(initial_h.begin(), initial_h.end(), 0.);

   RTensor<T> initial_c({num_directions, batch_size, hidden_size});
   std::fill(initial_c.begin(), initial_c.end(), 0.);

   RTensor<T> p({num_directions, 3 * hidden_size});
   std::fill(p.begin(), p.end(), 0.1);

   RTensor<T> y({seq_length, num_directions, batch_size, hidden_size});
   RTensor<T> y_h({num_directions, batch_size, hidden_size});
   RTensor<T> y_c({});

   T true_y_data[seq_length * num_directions * batch_size * hidden_size] = {
      0.37506911, 0.37506911, 0.37506911,
      0.6801309,  0.6801309,  0.6801309};
   RTensor<T> true_y(true_y_data, {seq_length, num_directions, batch_size, hidden_size});

   T true_y_h_data[num_directions * batch_size * hidden_size] = {
      0.37506911, 0.37506911, 0.37506911,
      0.6801309,  0.6801309,  0.6801309};
   RTensor<T> true_y_h(true_y_h_data, {num_directions, batch_size, hidden_size});

   ROperatorLSTM<T> lstm({{}, {}, {}, 0.0, "forward", hidden_size, 0, 0});
   lstm.Forward_blas(x, w, r, b, sequance_lens, initial_h, initial_c, p, y, y_h, y_c);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol);
   std::cout << "   Peepholes LSTM : Test ";
   std::cout << (failed? "Failed" : "Passed") << std::endl;
   return failed;
}


template<typename T>
bool testROperatorLSTM_batchwise(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorLSTM;
   const size_t seq_length = 1;
   const size_t batch_size = 3;
   const size_t input_size = 2;
   const size_t hidden_size = 7;
   const size_t num_directions = 1;

   RTensor<T> x({batch_size, seq_length, input_size});
   std::iota(x.begin(), x.end(), 1.);

   RTensor<T> w({num_directions, 4*hidden_size, input_size});
   std::fill(w.begin(), w.end(), 0.3);

   RTensor<T> b({});

   RTensor<T> r({num_directions, 4*hidden_size, hidden_size});
   std::fill(r.begin(), r.end(), 0.3);

   RTensor<size_t> sequance_lens({});

   RTensor<T> initial_h({});

   RTensor<T> initial_c({});

   RTensor<T> p({});

   RTensor<T> y({batch_size, seq_length, num_directions, hidden_size});
   RTensor<T> y_h({batch_size, num_directions, hidden_size});
   RTensor<T> y_c({});

   T true_y_data[batch_size * seq_length * num_directions * hidden_size] = {
      0.33369258, 0.33369258, 0.33369258, 0.33369258, 0.33369258, 0.33369258, 0.33369258,
      0.62239319, 0.62239319, 0.62239319, 0.62239319, 0.62239319, 0.62239319, 0.62239319,
 0.71857899, 0.71857899, 0.71857899, 0.71857899, 0.71857899, 0.71857899, 0.71857899};
   RTensor<T> true_y(true_y_data, {batch_size, seq_length, num_directions, hidden_size});

   T true_y_h_data[batch_size * num_directions * hidden_size] = {
      0.33369258, 0.33369258, 0.33369258, 0.33369258, 0.33369258, 0.33369258, 0.33369258,
      0.62239319, 0.62239319, 0.62239319, 0.62239319, 0.62239319, 0.62239319, 0.62239319,
      0.71857899, 0.71857899, 0.71857899, 0.71857899, 0.71857899, 0.71857899, 0.71857899};
   RTensor<T> true_y_h(true_y_h_data, {batch_size, num_directions, hidden_size});

   ROperatorLSTM<T> lstm({{}, {}, {}, 0.0, "forward", hidden_size, 0, 1});
   lstm.Forward_blas(x, w, r, b, sequance_lens, initial_h, initial_c, p, y, y_h, y_c);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol);
   std::cout << "   Batchwise LSTM : Test ";
   std::cout << (failed? "Failed" : "Passed") << std::endl;
   return failed;
}

template<typename T>
bool testROperatorLSTM_bidirectional(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorLSTM;
   const size_t seq_length = 3;
   const size_t batch_size = 1;
   const size_t input_size = 2;
   const size_t hidden_size = 3;
   const size_t num_directions = 2;

   RTensor<T> x({seq_length, batch_size, input_size});
   std::iota(x.begin(), x.end(), 1.);

   RTensor<T> w({num_directions, 4*hidden_size, input_size});
   std::fill(w.begin(), w.end(), 0.1);

   RTensor<T> b({});

   RTensor<T> r({num_directions, 4*hidden_size, hidden_size});
   std::fill(r.begin(), r.end(), 0.1);

   RTensor<size_t> sequance_lens({});

   RTensor<T> initial_h({});

   RTensor<T> initial_c({});

   RTensor<T> p({});

   RTensor<T> y({seq_length, num_directions, batch_size, hidden_size});
   RTensor<T> y_h({num_directions, batch_size, hidden_size});
   RTensor<T> y_c({num_directions, batch_size, hidden_size});

   T true_y_data[seq_length * num_directions * batch_size * hidden_size] = {
      0.0952, 0.0952, 0.0952,
      0.4041, 0.4041, 0.4041,
      0.3287, 0.3287, 0.3287,
      0.4927, 0.4927, 0.4927,
      0.6004, 0.6004, 0.6004,
      0.4032, 0.4032, 0.4032};
   RTensor<T> true_y(true_y_data, {seq_length, num_directions, batch_size, hidden_size});

   T true_y_h_data[num_directions * batch_size * hidden_size] = {
      0.6004, 0.6004, 0.6004,
      0.4041, 0.4041, 0.4041};
   RTensor<T> true_y_h(true_y_h_data, {num_directions, batch_size, hidden_size});

   T true_y_c_data[num_directions * batch_size * hidden_size] = {
      1.0493, 1.0493, 1.0493,
      0.7970, 0.7970, 0.7970};
   RTensor<T> true_y_c(true_y_c_data, {num_directions, batch_size, hidden_size});

   ROperatorLSTM<T> lstm({{}, {}, {}, 0.0, "bidirectional", hidden_size, 0, 0});
   lstm.Forward_blas(x, w, r, b, sequance_lens, initial_h, initial_c, p, y, y_h, y_c);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol) || !IsApprox(y_c, true_y_c, tol);
   std::cout << "   LSTM bidirectional : Test ";
   std::cout << (failed? "Failed" : "Passed") << std::endl;
   return failed;
}


#endif
