#ifndef TEST_ROPERATOR_GRU
#define TEST_ROPERATOR_GRU

#include "TMVA/RTensor.hxx"
#include "testROperator.hxx"
#include "ROperatorGRU.hxx"

#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>

template<typename T>
bool testROperatorGRU_defaults(double tol);

template<typename T>
bool testROperatorGRU_initial_bias(double tol);

template<typename T>
bool testROperatorGRU_seq_length(double tol);

template<typename T>
bool testROperatorGRU_batchwise(double tol);

template<typename T>
bool testROperatorGRU_bidirectional(double tol);

template<typename T>
bool testROperatorGRU(double tol) {
   bool failed = false;

   failed |= testROperatorGRU_defaults<T>(tol);
   failed |= testROperatorGRU_initial_bias<T>(tol);
   failed |= testROperatorGRU_seq_length<T>(tol);
   failed |= testROperatorGRU_batchwise<T>(tol);
   failed |= testROperatorGRU_bidirectional<T>(tol);

   return failed;
}

template<typename T>
bool testROperatorGRU_defaults(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorGRU;
   const size_t seq_length = 1;
   const size_t batch_size = 3;
   const size_t input_size = 2;
   const size_t hidden_size = 5;
   const size_t num_directions = 1;

   RTensor<T> x({seq_length, batch_size, input_size});
   std::iota(x.begin(), x.end(), 1.);

   RTensor<T> w({1, 3 * hidden_size, input_size});
   std::fill(w.begin(), w.end(), 0.1);

   RTensor<T> r({1, 3 * hidden_size, hidden_size});
   std::fill(r.begin(), r.end(), 0.1);

   RTensor<T> b({});

   RTensor<size_t> sequence_lens({});

   RTensor<T> initial_h({});

   RTensor<T> y({seq_length, num_directions , batch_size, hidden_size});
   RTensor<T> y_h({num_directions, batch_size, hidden_size});

   T true_y_data[seq_length * num_directions * batch_size * hidden_size] = {
      0.12397026, 0.12397026, 0.12397026, 0.12397026, 0.12397026,
      0.20053664, 0.20053664, 0.20053664, 0.20053664, 0.20053664,
      0.19991654, 0.19991654, 0.19991654, 0.19991654, 0.19991654
   };
   RTensor<T> true_y(true_y_data, {seq_length, num_directions, batch_size, hidden_size});

   T true_y_h_data[num_directions * batch_size * hidden_size] = {
      0.12397026, 0.12397026, 0.12397026, 0.12397026, 0.12397026,
      0.20053664, 0.20053664, 0.20053664, 0.20053664, 0.20053664,
      0.19991654, 0.19991654, 0.19991654, 0.19991654, 0.19991654
   };
   RTensor<T> true_y_h(true_y_h_data, {num_directions, batch_size, hidden_size});

   ROperatorGRU<T> gru({{}, {}, {}, 0., "forward", hidden_size, 0, 0});
   gru.Forward_blas(x, w, r, b, sequence_lens, initial_h, y, y_h);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol);
   std::cout << "   forward GRU : Test ";
   std::cout << (failed? "Failed" : "Passed" ) << std::endl;
   return failed;
}

template<typename T>
bool testROperatorGRU_initial_bias(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorGRU;
   const size_t seq_length = 1;
   const size_t batch_size = 3;
   const size_t input_size = 3;
   const size_t hidden_size = 3;
   const size_t num_directions = 1;

   RTensor<T> x({seq_length, batch_size, input_size});
   std::iota(x.begin(), x.end(), 1.);

   RTensor<T> w({1, 3 * hidden_size, input_size});
   std::fill(w.begin(), w.end(), 0.1);

   RTensor<T> r({1, 3 * hidden_size, hidden_size});
   std::fill(r.begin(), r.end(), 0.1);

   T b_data[6 * hidden_size] = {
      1., 1., 1., 1., 1., 1., 1., 1., 1.,
      0., 0., 0., 0., 0., 0., 0., 0., 0.};
   RTensor<T> b(b_data, {1, 6 * hidden_size});

   RTensor<size_t> sequence_lens({});

   RTensor<T> initial_h({});

   RTensor<T> y({seq_length, num_directions , batch_size, hidden_size});
   RTensor<T> y_h({num_directions, batch_size, hidden_size});

   T true_y_data[seq_length * num_directions * batch_size * hidden_size] = {
      0.15482332, 0.15482332, 0.15482332,
      0.0748427,  0.0748427,  0.0748427,
      0.0322236,  0.0322236,  0.0322236};
   RTensor<T> true_y(true_y_data, {seq_length, num_directions, batch_size, hidden_size});

   T true_y_h_data[num_directions * batch_size * hidden_size] = {
      0.15482332, 0.15482332, 0.15482332,
      0.0748427,  0.0748427,  0.0748427,
      0.0322236,  0.0322236,  0.0322236};
   RTensor<T> true_y_h(true_y_h_data, {num_directions, batch_size, hidden_size});

   ROperatorGRU<T> gru({{}, {}, {}, 0., "forward", hidden_size, 0, 0});
   gru.Forward_blas(x, w, r, b, sequence_lens, initial_h, y, y_h);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol);
   std::cout << "   GRU with initial bias : Test ";
   std::cout << (failed? "Failed" : "Passed" ) << std::endl;
   return failed;
}

template<typename T>
bool testROperatorGRU_seq_length(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorGRU;
   const size_t seq_length = 2;
   const size_t batch_size = 3;
   const size_t input_size = 3;
   const size_t hidden_size = 5;
   const size_t num_directions = 1;

   RTensor<T> x({seq_length, batch_size, input_size});
   std::iota(x.begin(), x.end(), 1.);

   RTensor<T> w({1, 3 * hidden_size, input_size});
   std::fill(w.begin(), w.end(), 0.1);

   RTensor<T> r({1, 3 * hidden_size, hidden_size});
   std::fill(r.begin(), r.end(), 0.1);

   T b_data[6 * hidden_size] = {
      1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
   RTensor<T> b(b_data, {1, 6 * hidden_size});

   RTensor<size_t> sequence_lens({});

   RTensor<T> initial_h({});

   RTensor<T> y({seq_length, num_directions , batch_size, hidden_size});
   RTensor<T> y_h({num_directions, batch_size, hidden_size});

   T true_y_data[seq_length * num_directions * batch_size * hidden_size] = {
      0.15482332, 0.15482332, 0.15482332, 0.15482332, 0.15482332,
      0.0748427,  0.0748427,  0.0748427,  0.0748427,  0.0748427,
      0.0322236,  0.0322236,  0.0322236,  0.0322236,  0.0322236,
      0.16530132, 0.16530132, 0.16530132, 0.16530132, 0.16530132,
      0.07973265, 0.07973265, 0.07973265, 0.07973265, 0.07973265,
      0.03435477, 0.03435477, 0.03435477, 0.03435477, 0.03435477};
   RTensor<T> true_y(true_y_data, {seq_length, num_directions, batch_size, hidden_size});

   T true_y_h_data[num_directions * batch_size * hidden_size] = {
      0.16530132, 0.16530132, 0.16530132, 0.16530132, 0.16530132,
      0.07973265, 0.07973265, 0.07973265, 0.07973265, 0.07973265,
      0.03435477, 0.03435477, 0.03435477, 0.03435477, 0.03435477};
   RTensor<T> true_y_h(true_y_h_data, {num_directions, batch_size, hidden_size});

   ROperatorGRU<T> gru({{}, {}, {}, 0., "forward", hidden_size, 0, 0});
   gru.Forward_blas(x, w, r, b, sequence_lens, initial_h, y, y_h);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol);
   std::cout << "   GRU with seq_length > 1 : Test ";
   std::cout << (failed? "Failed" : "Passed" ) << std::endl;
   return failed;
}

template<typename T>
bool testROperatorGRU_batchwise(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorGRU;
   const size_t seq_length = 1;
   const size_t batch_size = 3;
   const size_t input_size = 2;
   const size_t hidden_size = 6;
   const size_t num_directions = 1;

   RTensor<T> x({batch_size, seq_length, input_size});
   std::iota(x.begin(), x.end(), 1.);

   RTensor<T> w({1, 3 * hidden_size, input_size});
   std::fill(w.begin(), w.end(), 0.2);

   RTensor<T> r({1, 3 * hidden_size, hidden_size});
   std::fill(r.begin(), r.end(), 0.2);

   RTensor<T> b({});

   RTensor<size_t> sequence_lens({});

   RTensor<T> initial_h({});

   RTensor<T> y({batch_size, seq_length, num_directions, hidden_size});
   RTensor<T> y_h({batch_size, num_directions, hidden_size});

   T true_y_data[batch_size * seq_length * num_directions * hidden_size] = {
      0.19030017, 0.19030017, 0.19030017, 0.19030017, 0.19030017, 0.19030017,
      0.17513685, 0.17513685, 0.17513685, 0.17513685, 0.17513685, 0.17513685,
      0.09733084, 0.09733084, 0.09733084, 0.09733084, 0.09733084, 0.09733084};
   RTensor<T> true_y(true_y_data, {batch_size, seq_length, num_directions, hidden_size});

   T true_y_h_data[batch_size * num_directions * hidden_size] = {
      0.19030017, 0.19030017, 0.19030017, 0.19030017, 0.19030017, 0.19030017,
      0.17513685, 0.17513685, 0.17513685, 0.17513685, 0.17513685, 0.17513685,
      0.09733084, 0.09733084, 0.09733084, 0.09733084, 0.09733084, 0.09733084};
   RTensor<T> true_y_h(true_y_h_data, {batch_size, num_directions, hidden_size});

   ROperatorGRU<T> gru({{}, {}, {}, 0., "forward", hidden_size, 1, 0});
   gru.Forward_blas(x, w, r, b, sequence_lens, initial_h, y, y_h);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol);
   std::cout << "   batchwise GRU : Test ";
   std::cout << (failed? "Failed" : "Passed" ) << std::endl;
   return failed;
}

template<typename T>
bool testROperatorGRU_bidirectional(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorGRU;
   const size_t seq_length = 1;
   const size_t batch_size = 3;
   const size_t input_size = 2;
   const size_t hidden_size = 5;
   const size_t num_directions = 2;

   RTensor<T> x({seq_length, batch_size, input_size});
   std::iota(x.begin(), x.end(), 1.);

   RTensor<T> w({2, 3 * hidden_size, input_size});
   std::fill(w.begin(), w.begin() + 3 * hidden_size * input_size, 0.1);
   std::fill(w.begin() + 3 * hidden_size * input_size, w.end(), 0.01);

   RTensor<T> r({2, 3 * hidden_size, hidden_size});
   std::fill(r.begin(), r.end(), 0.1);

   RTensor<T> b({});

   RTensor<size_t> sequence_lens({});

   RTensor<T> initial_h({});

   RTensor<T> y({seq_length, num_directions , batch_size, hidden_size});
   RTensor<T> y_h({num_directions, batch_size, hidden_size});

   T true_y_data[seq_length * num_directions * batch_size * hidden_size] = {
      0.12397026, 0.12397026, 0.12397026, 0.12397026, 0.12397026,
      0.20053664, 0.20053664, 0.20053664, 0.20053664, 0.20053664,
      0.19991654, 0.19991654, 0.19991654, 0.19991654, 0.19991654,
      0.01477059, 0.01477059, 0.01477059, 0.01477059, 0.01477059,
      0.03372044, 0.03372044, 0.03372044, 0.03372044, 0.03372044,
      0.05176941, 0.05176941, 0.05176941, 0.05176941, 0.05176941};
   RTensor<T> true_y(true_y_data, {seq_length, num_directions, batch_size, hidden_size});

   T true_y_h_data[num_directions * batch_size * hidden_size] = {
      0.12397026, 0.12397026, 0.12397026, 0.12397026, 0.12397026,
      0.20053664, 0.20053664, 0.20053664, 0.20053664, 0.20053664,
      0.19991654, 0.19991654, 0.19991654, 0.19991654, 0.19991654,
      0.01477059, 0.01477059, 0.01477059, 0.01477059, 0.01477059,
      0.03372044, 0.03372044, 0.03372044, 0.03372044, 0.03372044,
      0.05176941, 0.05176941, 0.05176941, 0.05176941, 0.05176941};
   RTensor<T> true_y_h(true_y_h_data, {num_directions, batch_size, hidden_size});

   ROperatorGRU<T> gru({{}, {}, {}, 0., "bidirectional", hidden_size, 0, 0});
   gru.Forward_blas(x, w, r, b, sequence_lens, initial_h, y, y_h);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol);
   std::cout << "   bidirectional GRU: Test ";
   std::cout << (failed? "Failed" : "Passed" ) << std::endl;
   return failed;
}


#endif
