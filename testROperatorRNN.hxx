#ifndef TEST_ROPERATOR_RNN
#define TEST_ROPERATOR_RNN

#include "TMVA/RTensor.hxx"
#include "testROperator.hxx"
#include "ROperatorRNN.hxx"

#include <algorithm>
#include <iostream>
#include <random>
#include <sstream>

template<typename T>
bool testROperatorRNN_bidirectional(double tol);

template<typename T>
bool testROperatorRNN_bidirectional_batchwise(double tol);

template<typename T>
bool testROperatorRNN_forward(double tol);

template<typename T>
bool testROperatorRNN_backward(double tol);

template<typename T>
bool testROperatorRNN_sequence(double tol);

template<typename T>
bool testROperatorRNN_sequence_batchwise(double tol);

template<typename T>
bool testROperatorRNN(double tol) {
   bool failed = false;

   failed |= testROperatorRNN_bidirectional<T>(tol);
   failed |= testROperatorRNN_bidirectional_batchwise<T>(tol);
   failed |= testROperatorRNN_forward<T>(tol);
   failed |= testROperatorRNN_backward<T>(tol);
   failed |= testROperatorRNN_sequence<T>(tol);
   failed |= testROperatorRNN_sequence_batchwise<T>(tol);

   return failed;
}

template<typename T>
bool testROperatorRNN_bidirectional(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorRNN;
   const size_t seq_length = 3;
   const size_t batch_size = 3;
   const size_t input_size = 2;
   const size_t hidden_size = 4;
   const size_t num_directions = 2;

   T x_data[seq_length * batch_size * input_size] = {
      0,    0.01, 0.02, 0.03, 0.04, 0.05,
      0.06, 0.07, 0.08, 0.09, 0.1,  0.11,
      0.12, 0.13, 0.14, 0.15, 0.16, 0.17};
   RTensor<T> x(x_data, {seq_length, batch_size, input_size});

   T w_data[num_directions * hidden_size * input_size]= {
      1.16308,    2.21221,
      0.483805,   0.774004,
      0.299563,   1.04344,
      0.153025,   1.18393,
      -1.16881,   1.89171,
      1.55807,   -1.23474,
      -0.545945, -1.77103,
      -2.35563,  -0.451384};
   RTensor<T> w(w_data, {num_directions, hidden_size, input_size});

   T r_data[num_directions * hidden_size * hidden_size] = {
      -0.264848, -1.30311,   0.0712087,  0.64198,
      -2.76538,  -0.652074, -0.784275,  -1.76749,
      -0.450673, -0.917929, -0.966654,   0.650856,
      0.285538,  -0.909848, -1.90459,   -0.140926,
      -1.37131,   0.780644,  0.441009,   1.15856,
      0.313298,   1.96766,  -1.11991,   -0.00440959,
      0.407622,   2.60569,  -0.840986,   0.585658,
      0.823292,  -0.696818,  1.15115,    0.150269};
   RTensor<T> r(r_data, {num_directions, hidden_size, hidden_size});

   T b_data[num_directions * 2*hidden_size] = {
      -0.161029, -2.58991, 0.339721,  -0.31664,   0.049053, -1.89795, -0.327121, -0.159628,
      -0.183054, -0.977459, -1.08309, -0.0165881, 1.99349,   1.35513, -0.697978, -0.708618};
   RTensor<T> b(b_data, {num_directions, 2*hidden_size});

   RTensor<size_t> sequence_lens({});

   T initial_h_data[num_directions * batch_size * hidden_size] = {
      -0.371075, 0.252533, -1.42195, 0.39303,
      -0.463112, -1.02438, -0.538399, -2.21508,
      -1.4221, -0.149365, 1.2587, 1.38294,
      -0.0841612, 1.45697, 0.0679387, 2.11548,
      -1.51051, 1.50948, 0.206351, -0.981445,
      -0.221477, -0.230484, 0.453313, 0.795476};
   RTensor<T> initial_h(initial_h_data, {num_directions, batch_size, hidden_size});

   RTensor<T> y({seq_length, num_directions , batch_size, hidden_size});
   RTensor<T> y_h({num_directions, batch_size, hidden_size});

   T true_y_data[seq_length * num_directions * batch_size * hidden_size] = {
      -0.1680, -0.9967,  0.9200,  0.9520,
      -0.0252,  0.9499,  0.2707,  0.9354,
      0.9207,  -0.9991,  0.4916, -0.9971,
      0.9570,   0.9907,  0.8348, -0.8432,
      0.9337,   0.9806,  0.4609, -0.8339,
      0.9651,   0.9867, -0.1637, -0.9962,
      0.9723,  -1.0000,  0.6775, -0.8878,
      -0.4064, -1.0000, -0.3654, -0.9542,
      0.6047,  -0.9999, -0.4339,  0.0454,
      0.5554,   0.9138, -0.3105,  0.2289,
      0.3086,   0.9752,  0.0683, -0.4540,
      -0.7255,  0.6310, -0.9871, -0.7722,
      0.6853,  -1.0000, -0.5145, -0.2746,
      0.8194,  -0.5044,  0.7796,  0.8733,
      0.9203,  -0.9999,  0.8698,  0.9291,
      1.0000,   0.9964,  0.9936, -0.9419,
      0.9995,   0.9907,  0.4186, -0.9974,
      0.9966,  -0.5491, -0.9923, -0.5074
   };
   RTensor<T> true_y(true_y_data, {seq_length, num_directions, batch_size, hidden_size});

   T true_y_h_data[num_directions * batch_size * hidden_size] = {
      0.68528736, -0.99995285, -0.51453173, -0.27458954,
      0.81935215, -0.5043667,   0.7795707,   0.8733188,
      0.92034006, -0.999916,    0.8697902,   0.9291247,
      0.9570278,   0.990718,    0.83482444, -0.8432297,
      0.9336523,   0.98059773,  0.46091712, -0.83394974,
      0.9651197,   0.98665035, -0.16370827, -0.99619436
   };
   RTensor<T> true_y_h(true_y_h_data, {num_directions, batch_size, hidden_size});

   ROperatorRNN<T> rnn({{}, {}, {}, 0.0, "bidirectional", hidden_size, 0});
   rnn.Forward_blas(x, w, r, b, sequence_lens, initial_h, y, y_h);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol);
   std::cout << "   Bidirectional RNN : Test ";
   std::cout << (failed? "Failed" : "Passed" ) << std::endl;
   return failed;
}


template<typename T>
bool testROperatorRNN_bidirectional_batchwise(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorRNN;
   const size_t seq_length = 3;
   const size_t batch_size = 3;
   const size_t input_size = 2;
   const size_t hidden_size = 4;
   const size_t num_directions = 2;

   T x_data[batch_size * seq_length * input_size] = {
      0,    0.01, 0.06, 0.07, 0.12, 0.13,
      0.02, 0.03, 0.08, 0.09, 0.14, 0.15,
      0.04, 0.05, 0.1,  0.11, 0.16, 0.17};
   RTensor<T> x(x_data, {batch_size, seq_length, input_size});

   T w_data[num_directions * hidden_size * input_size]= {
      1.16308,    2.21221,
      0.483805,   0.774004,
      0.299563,   1.04344,
      0.153025,   1.18393,
      -1.16881,   1.89171,
      1.55807,   -1.23474,
      -0.545945, -1.77103,
      -2.35563,  -0.451384};
   RTensor<T> w(w_data, {num_directions, hidden_size, input_size});

   T r_data[num_directions * hidden_size * hidden_size] = {
      -0.264848, -1.30311,   0.0712087,  0.64198,
      -2.76538,  -0.652074, -0.784275,  -1.76749,
      -0.450673, -0.917929, -0.966654,   0.650856,
      0.285538,  -0.909848, -1.90459,   -0.140926,
      -1.37131,   0.780644,  0.441009,   1.15856,
      0.313298,   1.96766,  -1.11991,   -0.00440959,
      0.407622,   2.60569,  -0.840986,   0.585658,
      0.823292,  -0.696818,  1.15115,    0.150269};
   RTensor<T> r(r_data, {num_directions, hidden_size, hidden_size});

   T b_data[num_directions * 2*hidden_size] = {
      -0.161029, -2.58991, 0.339721,  -0.31664,   0.049053, -1.89795, -0.327121, -0.159628,
      -0.183054, -0.977459, -1.08309, -0.0165881, 1.99349,   1.35513, -0.697978, -0.708618};
   RTensor<T> b(b_data, {num_directions, 2*hidden_size});

   RTensor<size_t> sequence_lens({});

   T initial_h_data[batch_size * num_directions * hidden_size] = {
      -0.371075,   0.252533, -1.42195,    0.39303,
      -0.0841612,  1.45697,   0.0679387,  2.11548,
      -0.463112,  -1.02438,  -0.538399,  -2.21508,
      -1.51051,    1.50948,   0.206351,  -0.981445,
      -1.4221,    -0.149365,  1.2587,     1.38294,
      -0.221477,  -0.230484,  0.453313,   0.795476};
   RTensor<T> initial_h(initial_h_data, {batch_size, num_directions, hidden_size});

   T true_y_data[batch_size * seq_length * num_directions * hidden_size] = {
      -0.168,  -0.9967,  0.92,     0.952,
       0.957,   0.9907,  0.8348,  -0.8432,
       0.9723, -1,       0.6775,  -0.8878,
       0.5554,  0.9138, -0.3105,   0.2289,
       0.6853, -1,      -0.5145,  -0.2746,
       1,       0.9964,  0.9936,  -0.9419,
      -0.0252,  0.9499,  0.2707,   0.9354,
       0.9337,  0.9806,  0.4609,  -0.8339,
      -0.4064, -1,       -0.3654, -0.9542,
       0.3086,  0.9752,  0.0683,  -0.454,
       0.8194, -0.5044,  0.7796,   0.8733,
       0.9995,  0.9907,  0.4186,  -0.9974,
       0.9207, -0.9991,  0.4916,  -0.9971,
       0.9651,  0.9867, -0.1637,  -0.9962,
       0.6047, -0.9999, -0.4339,   0.0454,
      -0.7255,  0.631,  -0.9871,  -0.7722,
       0.9203, -0.9999,  0.8698,   0.9291,
       0.9966, -0.5491, -0.9923,  -0.5074};
   RTensor<T> true_y(true_y_data, {batch_size, seq_length, num_directions, hidden_size});

   T true_y_h_data[batch_size * num_directions * hidden_size] = {
      0.685287, -0.999953, -0.514532, -0.27459,
      0.957028,  0.990718,  0.834824, -0.84323,
      0.819352, -0.504367,  0.779571,  0.873319,
      0.933652,  0.980598,  0.460917, -0.83395,
      0.92034,  -0.999916,  0.86979,   0.929125,
      0.96512,   0.98665,  -0.163708, -0.996194};
   RTensor<T> true_y_h(true_y_h_data, {batch_size, num_directions, hidden_size});

   RTensor<T> y({batch_size, seq_length, num_directions, hidden_size});
   RTensor<T> y_h({batch_size, num_directions, hidden_size});

   ROperatorRNN<T> rnn({{}, {}, {}, 0.0, "bidirectional", hidden_size, 1});
   rnn.Forward_blas(x, w, r, b, sequence_lens, initial_h, y, y_h);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol);
   std::cout << "   Batchwise Bidirectional RNN : Test ";
   std::cout << (failed? "Failed" : "Passed" ) << std::endl;

   return failed;
}

template<typename T>
bool testROperatorRNN_forward(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorRNN;
   const size_t seq_length = 2;
   const size_t batch_size = 3;
   const size_t input_size = 3;
   const size_t hidden_size = 5;

   T x_data[seq_length * batch_size * input_size] = {
      0.01, 0.02, 0.03,
      0.04, 0.05, 0.06,
      0.07, 0.08, 0.09,
      0.1 , 0.11, 0.12,
      0.13, 0.14, 0.15,
      0.16, 0.17, 0.18};
   RTensor<T> x(x_data, {seq_length, batch_size, input_size});

   T w_data[hidden_size * input_size] = {
       0.17244112,  0.43839353, -0.8767394,
      -0.7451854 ,  0.95555747,  0.57059574,
       0.8873628 ,  0.41318333,  0.9505067,
       0.39907658, -0.9042908 ,  0.32292694,
       2.1221848 ,  0.24675833, -0.5175528};
   RTensor<T> w(w_data, {1, hidden_size, input_size});

   T r_data[hidden_size * hidden_size] = {
      -0.3538616 , -1.7107042 , -0.57643396,  0.71042347, -1.6298728,
       1.646922  ,  0.0743815 ,  1.600358  , -0.43490064, -1.2173619,
      -1.0533832 ,  0.27596667, -1.8260423 , -0.6393682 ,  0.8086523,
       1.4254899 ,  1.0429248 , -0.6465876 ,  1.5841872 , -0.8080765,
      -1.0693161 ,  0.53074974, -1.1467112 , -0.62543255,  0.8353564
   };
   RTensor<T> r(r_data, {1, hidden_size, hidden_size});

   T b_data[2 * hidden_size] = {
      1.0725956,  0.04914486, -0.40236193, -0.5165786,  1.8941225 ,
      0.69393295, 0.11429706, -1.6716264,  -2.2173078, -1.76375};
   RTensor<T> b(b_data, {1, 2 * hidden_size});

   RTensor<size_t> sequence_lens({});
   RTensor<T> initial_h({});

   RTensor<T> y({seq_length, 1 , batch_size, hidden_size});
   RTensor<T> y_h({1, batch_size, hidden_size});

   T true_y_data[seq_length * batch_size * hidden_size] = {
      0.9415,  0.1899, -0.9660, -0.9917,  0.1401,
      0.9405,  0.2124, -0.9612, -0.9918,  0.1941,
      0.9396,  0.2346, -0.9557, -0.9918,  0.2469,
      0.5998,  0.4944, -0.2485, -0.9791,  0.8473,
      0.5049,  0.4680, -0.1433, -0.9803,  0.8751,
      0.3976,  0.4427, -0.0368, -0.9814,  0.8978};
   RTensor<T> true_y(true_y_data, {seq_length, 1 , batch_size, hidden_size});

   T true_y_h_data[batch_size * hidden_size] = {
      0.5998,  0.4944, -0.2485, -0.9791,  0.8473,
      0.5049,  0.4680, -0.1433, -0.9803,  0.8751,
      0.3976,  0.4427, -0.0368, -0.9814,  0.8978};
   RTensor<T> true_y_h(true_y_h_data, {1, batch_size, hidden_size});

   ROperatorRNN<T> rnn({{}, {}, {}, 0.0, "forward", hidden_size, 0});
   rnn.Forward_blas(x, w, r, b, sequence_lens, initial_h, y, y_h);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol);
   std::cout << "   Forward RNN : Test ";
   std::cout << (failed? "Failed" : "Passed" ) << std::endl;

   return failed;
}

template<typename T>
bool testROperatorRNN_backward(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorRNN;
   const size_t seq_length = 2;
   const size_t batch_size = 3;
   const size_t input_size = 3;
   const size_t hidden_size = 5;

   T x_data[seq_length * batch_size * input_size] = {
      0.01, 0.02, 0.03,
      0.04, 0.05, 0.06,
      0.07, 0.08, 0.09,
      0.1 , 0.11, 0.12,
      0.13, 0.14, 0.15,
      0.16, 0.17, 0.18};
   RTensor<T> x(x_data, {seq_length, batch_size, input_size});

   T w_data[hidden_size * input_size] = {
       0.17244112,  0.43839353, -0.8767394,
      -0.7451854 ,  0.95555747,  0.57059574,
       0.8873628 ,  0.41318333,  0.9505067,
       0.39907658, -0.9042908 ,  0.32292694,
       2.1221848 ,  0.24675833, -0.5175528};
   RTensor<T> w(w_data, {1, hidden_size, input_size});

   T r_data[hidden_size * hidden_size] = {
      -0.3538616 , -1.7107042 , -0.57643396,  0.71042347, -1.6298728,
       1.646922  ,  0.0743815 ,  1.600358  , -0.43490064, -1.2173619,
      -1.0533832 ,  0.27596667, -1.8260423 , -0.6393682 ,  0.8086523,
       1.4254899 ,  1.0429248 , -0.6465876 ,  1.5841872 , -0.8080765,
      -1.0693161 ,  0.53074974, -1.1467112 , -0.62543255,  0.8353564
   };
   RTensor<T> r(r_data, {1, hidden_size, hidden_size});

   T b_data[2 * hidden_size] = {
      1.0725956,  0.04914486, -0.40236193, -0.5165786,  1.8941225 ,
      0.69393295, 0.11429706, -1.6716264,  -2.2173078, -1.76375};
   RTensor<T> b(b_data, {1, 2 * hidden_size});

   RTensor<size_t> sequence_lens({});
   RTensor<T> initial_h({});

   RTensor<T> y({seq_length, 1 , batch_size, hidden_size});
   RTensor<T> y_h({1, batch_size, hidden_size});

   T true_y_data[seq_length * batch_size * hidden_size] = {
      0.3238,  0.2968, -0.3250, -0.9813,  0.8431,
      0.2030,  0.2733, -0.2311, -0.9823,  0.8700,
      0.0786,  0.2529, -0.1361, -0.9833,  0.8921,
      0.9387,  0.2566, -0.9494, -0.9919,  0.2983,
      0.9377,  0.2784, -0.9423, -0.9920,  0.3480,
      0.9367,  0.2999, -0.9342, -0.9921,  0.3958};
   RTensor<T> true_y(true_y_data, {seq_length, 1 , batch_size, hidden_size});

   T true_y_h_data[batch_size * hidden_size] = {
      0.3238,  0.2968, -0.3250, -0.9813,  0.8431,
      0.2030,  0.2733, -0.2311, -0.9823,  0.8700,
      0.0786,  0.2529, -0.1361, -0.9833,  0.8921};
   RTensor<T> true_y_h(true_y_h_data, {1, batch_size, hidden_size});

   ROperatorRNN<T> rnn({{}, {}, {}, 0.0, "backward", hidden_size, 0});
   rnn.Forward_blas(x, w, r, b, sequence_lens, initial_h, y, y_h);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol);
   std::cout << "   Backward RNN : Test ";
   std::cout << (failed? "Failed" : "Passed" ) << std::endl;

   return failed;
}


template<typename T>
bool testROperatorRNN_sequence(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorRNN;
   const size_t batch_size = 3;
   const size_t seq_length = 3;
   const size_t input_size = 5;
   const size_t hidden_size = 6;

   T x_data[seq_length * batch_size * input_size] = {
       0.01,  -0.01,   0.08,   0.09,    0.001,
       .09,   -0.7,   -0.35,   0.0,     0.001,
       0.16,  -0.19,   0.003,  0.,      0.0001,
       0.05,  -0.09,   0.013,  0.5,     0.005,
       .2,    -0.05,   .062,  -0.04,   -0.04,
       0.,     0.,     0.,     0.,      0.,
       0.06,   0.087,  0.01,   0.3,    -0.001,
       0.,     0.,     0.,     0.,      0.,
       0.,     0.,     0.,     0.,      0.};
   RTensor<T> x(x_data, {seq_length, batch_size, input_size});

   T w_data[hidden_size * input_size] = {
      0.2369,  0.1346,  0.3317, -0.4822, -0.1363,
      0.9420, -0.4502, -2.8174,  0.2889,  1.6715,
      0.2967,  1.6799, -0.8343,  0.4493,  0.0370,
      -0.5326,  1.1545, -1.6478,  0.7779, -0.9257,
      -1.4822, -0.8717, -0.0174,  2.0685, -0.7620,
      0.0105, -2.9378,  0.8887, -0.9478, -1.5725};
   RTensor<T> w(w_data, {1, hidden_size, input_size});

   T r_data[hidden_size * hidden_size] = {
      1.0135, -0.2632, -0.6786, -1.0179, -2.1319, -0.0036,
      1.9585,  1.1375,  2.1210,  0.6409, -2.0503, -2.4921,
      0.5932,  1.5161, -0.7769,  0.2849,  0.2072, -0.3086,
      -0.9655,  0.9178, -0.4292, -1.5054, -0.7396,  0.8929,
      -0.1836, -1.6292,  1.0712,  0.3770,  0.1779, -1.1167,
      -0.6861,  1.2391, -0.5448, -0.3881, -0.5165,  0.0128};
   RTensor<T> r(r_data, {1, hidden_size, hidden_size});

   RTensor<T> b({});

   RTensor<size_t> sequence_lens({batch_size});
   sequence_lens(0) = 3;
   sequence_lens(1) = 2;
   sequence_lens(2) = 1;

   RTensor<T> initial_h({});

   RTensor<T> y({seq_length, 1, batch_size, hidden_size});
   RTensor<T> y_h({1, batch_size, hidden_size});

   T true_y_data[seq_length * batch_size * hidden_size] = {
      -0.0160, -0.1818, -0.0401, -0.0794,  0.1761,  0.0137,
      -0.1869,  0.8827, -0.6948, -0.2732,  0.4479,  0.9408,
       0.0133,  0.2241, -0.2675, -0.3001, -0.0715,  0.5097,
      -0.4409, -0.5119, -0.1651,  0.0995,  0.8556, -0.4281,
      -0.4965, -0.9996,  0.8845,  0.9602, -0.9983,  0.9460,
       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
      -0.9776, -0.9818, -0.2740, -0.6920,  0.9529, -0.8501,
       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000};
   RTensor<T> true_y(true_y_data, {seq_length , 1, batch_size, hidden_size});

   T true_y_h_data[batch_size * hidden_size] = {
      -0.9776, -0.9818, -0.2740, -0.6920,  0.9529, -0.8501,
      -0.4965, -0.9996,  0.8845,  0.9602, -0.9983,  0.9460,
      0.0133,  0.2241, -0.2675, -0.3001, -0.0715,  0.5097};
   RTensor<T> true_y_h(true_y_h_data, {1, batch_size, hidden_size});

   ROperatorRNN<T> rnn({{}, {}, {}, 0.0, "forward", hidden_size, 0});
   rnn.Forward_blas(x, w, r, b, sequence_lens, initial_h, y, y_h);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol);
   std::cout << "   RNN with different sequence lengths : Test ";
   std::cout << (failed? "Failed" : "Passed" ) << std::endl;

   return failed;
}

template<typename T>
bool testROperatorRNN_sequence_batchwise(double tol) {
   using namespace TMVA::Experimental;
   using TMVA::Experimental::SOFIE::ROperatorRNN;
   const size_t batch_size = 3;
   const size_t seq_length = 3;
   const size_t input_size = 5;
   const size_t hidden_size = 6;

   T x_data[batch_size * seq_length * input_size] = {
       0.01,  -0.01,   0.08,   0.09,    0.001,
       0.05,  -0.09,   0.013,  0.5,     0.005,
       0.06,   0.087,  0.01,   0.3,    -0.001,
       .09,   -0.7,   -0.35,   0.0,     0.001,
       .2,    -0.05,   .062,  -0.04,   -0.04,
       0.,     0.,     0.,     0.,      0.,
       0.16,  -0.19,   0.003,  0.,      0.0001,
       0.,     0.,     0.,     0.,      0.,
       0.,     0.,     0.,     0.,      0.};
   RTensor<T> x(x_data, {seq_length, batch_size, input_size});

   T w_data[hidden_size * input_size] = {
      0.2369,  0.1346,  0.3317, -0.4822, -0.1363,
      0.9420, -0.4502, -2.8174,  0.2889,  1.6715,
      0.2967,  1.6799, -0.8343,  0.4493,  0.0370,
      -0.5326,  1.1545, -1.6478,  0.7779, -0.9257,
      -1.4822, -0.8717, -0.0174,  2.0685, -0.7620,
      0.0105, -2.9378,  0.8887, -0.9478, -1.5725};
   RTensor<T> w(w_data, {1, hidden_size, input_size});

   T r_data[hidden_size * hidden_size] = {
      1.0135, -0.2632, -0.6786, -1.0179, -2.1319, -0.0036,
      1.9585,  1.1375,  2.1210,  0.6409, -2.0503, -2.4921,
      0.5932,  1.5161, -0.7769,  0.2849,  0.2072, -0.3086,
      -0.9655,  0.9178, -0.4292, -1.5054, -0.7396,  0.8929,
      -0.1836, -1.6292,  1.0712,  0.3770,  0.1779, -1.1167,
      -0.6861,  1.2391, -0.5448, -0.3881, -0.5165,  0.0128};
   RTensor<T> r(r_data, {1, hidden_size, hidden_size});

   RTensor<T> b({});

   RTensor<size_t> sequence_lens({batch_size});
   sequence_lens(0) = 3;
   sequence_lens(1) = 2;
   sequence_lens(2) = 1;

   RTensor<T> initial_h({});

   RTensor<T> y({batch_size, seq_length, 1, hidden_size});
   RTensor<T> y_h({batch_size, 1, hidden_size});

   T true_y_data[batch_size * seq_length * hidden_size] = {
      -0.0160, -0.1818, -0.0401, -0.0794,  0.1761,  0.0137,
      -0.4409, -0.5119, -0.1651,  0.0995,  0.8556, -0.4281,
      -0.9776, -0.9818, -0.2740, -0.6920,  0.9529, -0.8501,
      -0.1869,  0.8827, -0.6948, -0.2732,  0.4479,  0.9408,
      -0.4965, -0.9996,  0.8845,  0.9602, -0.9983,  0.9460,
       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
       0.0133,  0.2241, -0.2675, -0.3001, -0.0715,  0.5097,
       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
       0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000};
   RTensor<T> true_y(true_y_data, {batch_size, seq_length , 1, hidden_size});

   T true_y_h_data[batch_size * hidden_size] = {
      -0.9776, -0.9818, -0.2740, -0.6920,  0.9529, -0.8501,
      -0.4965, -0.9996,  0.8845,  0.9602, -0.9983,  0.9460,
       0.0133,  0.2241, -0.2675, -0.3001, -0.0715,  0.5097};
   RTensor<T> true_y_h(true_y_h_data, {batch_size, 1, hidden_size});

   ROperatorRNN<T> rnn({{}, {}, {}, 0.0, "forward", hidden_size, 1});
   rnn.Forward_blas(x, w, r, b, sequence_lens, initial_h, y, y_h);

   bool failed = !IsApprox(y, true_y, tol) || !IsApprox(y_h, true_y_h, tol);
   std::cout << "   Batchwise RNN with different sequence lengths : Test ";
   std::cout << (failed? "Failed" : "Passed" ) << std::endl;

   return failed;
}

#endif
