#ifndef TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_LSTM
#define TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_LSTM

#include <TMVA/RTensor.hxx>
#include <iostream>
#include <stdexcept>
#include <string>

#include "Blas.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

// Operator LSTM
template<typename T> class ROperatorLSTM {
private:
   /* Attributes */
   std::vector<T> fActivationAlpha;
   std::vector<T> fActivationBeta;
   std::vector<std::string> fActivations;
   T fClip;
   std::string fDirection;
   std::size_t fHiddenSize;
   std::size_t fInputForget;
   std::size_t fLayout;

public:
   /* Constructor */
   ROperatorLSTM(std::vector<T> activationAlpha, std::vector<T> activationBeta,
      std::vector<std::string> activations, T clip, std::string direction,
      std::size_t hiddenSize, std::size_t inputForget, std::size_t layout):
      fActivationAlpha(activationAlpha), fActivationBeta(activationBeta),
      fActivations(activations), fClip(clip), fDirection(direction),
      fHiddenSize(hiddenSize), fInputForget(inputForget), fLayout(layout)
   {}

   /* Forward pass using Blas */
   void Forward_blas(RTensor<T> &X,
                     RTensor<T> &W,
                     RTensor<T> &R,
                     RTensor<T> &B,
                     RTensor<size_t> &sequence_lens,
                     RTensor<T> &initial_h,
                     RTensor<T> &initial_c,
                     RTensor<T> &P,
                     RTensor<T> &Y,
                     RTensor<T> &Y_h,
                     RTensor<T> &Y_c);

};

template<typename T>
void ROperatorLSTM<T>::Forward_blas(RTensor<T> &X,
                                    RTensor<T> &W,
                                    RTensor<T> &R,
                                    RTensor<T> &B,
                                    RTensor<size_t> &sequence_lens,
                                    RTensor<T> &initial_h,
                                    RTensor<T> &initial_c,
                                    RTensor<T> &P,
                                    RTensor<T> &Y,
                                    RTensor<T> &Y_h,
                                    RTensor<T> &Y_c) {
   // Activation functions
   if (fActivations.empty()) {
      if (fDirection == "forward" || fDirection == "backward") {
         fActivations = {"Sigmoid", "Tanh", "Tanh"};
      } else {
         fActivations = {"Sigmoid", "Tanh", "Tanh", "Sigmoid", "Tanh", "Tanh"};
      }
   }
   // Check the attributes
   for (auto &activation : fActivations) {
      if (activation != "Relu" && activation != "Tanh" &&
          activation != "Sigmoid" && activation != "Affine" &&
          activation != "LeakyRelu" && activation != "ThresholdRelu" &&
          activation != "ScaledTanh" && activation != "HardSigmoid" &&
          activation != "Elu" && activation != "Softsign" &&
          activation != "Softplus") {
         throw std::runtime_error("TMVA SOFIE - Activation function " +
                                  activation + " not implemented");
      }
   }
   if (fDirection != "forward" && fDirection != "backward"
      && fDirection != "bidirectional") {
      throw
         std::runtime_error("TMVA SOFIE - Invalid fDirection = " + fDirection);
   }
   if (4*fHiddenSize != W.GetShape()[1]) {
      throw std::runtime_error(
         "TMVA SOFIE - fHiddenSize must be equal to " + std::to_string(W.GetShape()[1]));
   }
   if (fInputForget > 1) {
      throw
         std::runtime_error("TMVA SOFIE - fInputForget=" + std::to_string(fInputForget)
            + " must be 0 or 1.");
   }
   if (fLayout > 1) {
      throw
         std::runtime_error("TMVA SOFIE - Invalid fLayout = " + std::to_string(fLayout));
   }

   size_t seq_length = (fLayout == 0) ? X.GetShape()[0] : X.GetShape()[1];
   size_t batch_size = (fLayout == 0) ? X.GetShape()[1] : X.GetShape()[0];
   size_t input_size = X.GetShape()[2];
   size_t num_directions = W.GetShape()[0];

   // Set the input
   T* input = nullptr;
   if (fLayout == 0) {
      input = X.GetData();
   } else {
      input = new T[seq_length * batch_size * input_size];
      for(size_t seq = 0; seq < seq_length; seq++) {
         for (size_t batch = 0; batch < batch_size; batch++) {
            for(size_t i = 0; i < input_size; i++) {
               input[seq * batch_size * input_size + batch * input_size + i] = X.GetData()[
                  batch * seq_length * input_size + seq * input_size + i];
            }
         }
      }
   }

   // Broadcasting the bias
   T* bias = nullptr;
   if (B.GetShape().size() > 0) {
      bias = new T[4 * num_directions * seq_length * batch_size * fHiddenSize];
      for (size_t gate = 0; gate < 4; gate++) {
         T sum[fHiddenSize];
         for (size_t direction = 0; direction < num_directions; direction++) {
            // Compute the sum of the gate-hidden bias and the hidden-hidden bias
            size_t offset = direction * 8 * fHiddenSize + gate * fHiddenSize;
            for (size_t h = 0; h < fHiddenSize; h++) {
               sum[h] = B.GetData()[offset + h] + B.GetData()[offset + h + 4 * fHiddenSize];
            }
            // Copy sum into bias
            for (size_t seq = 0; seq < seq_length; seq++) {
               for (size_t batch = 0; batch < batch_size; batch++) {
                  size_t bias_offset = gate * num_directions * seq_length * batch_size * fHiddenSize
                     + direction * seq_length * batch_size * fHiddenSize
                     + seq * batch_size * fHiddenSize + batch * fHiddenSize;
                  std::copy(sum, sum + fHiddenSize, bias + bias_offset);
               }
            }
         }
      }
   }
   // Broadcasting the weight for peepholes
   T* peephole = nullptr;
   if (P.GetShape().size() > 0) {
      if (batch_size == 1) {
         peephole = P.GetData();
      } else {
         peephole = new T[num_directions * 3 * batch_size * fHiddenSize];
         for (size_t direction = 0; direction < num_directions; direction++) {
            for (size_t gate = 0; gate < 3; gate++) {
               size_t p_offset = direction * 3 * fHiddenSize + gate * fHiddenSize;
               for (size_t batch = 0; batch < batch_size; batch++) {
                  size_t offset = direction * 3 * batch_size * fHiddenSize +
                     gate * batch_size * fHiddenSize + batch * fHiddenSize;
                  std::copy(P.GetData() + p_offset, P.GetData() + p_offset + fHiddenSize,
                     peephole + offset);
               }
            }
         }
      }
   }
   // Set the initial hidden state
   T* initial_hidden_state = nullptr;
   if (initial_h.GetShape().size() > 0) {
      if (fLayout == 0) {
         initial_hidden_state = initial_h.GetData();
      } else {
         initial_hidden_state = new T[num_directions * batch_size * fHiddenSize];
         for (size_t direction = 0; direction < num_directions; direction++) {
            for (size_t batch = 0; batch < batch_size; batch++) {
               for (size_t h = 0; h < fHiddenSize; h++) {
                  initial_hidden_state[direction * batch_size * fHiddenSize + batch * fHiddenSize + h] =
                     initial_h.GetData()[batch * num_directions * fHiddenSize + direction * fHiddenSize
                        + h];
               }
            }
         }
      }
   }
   // Set the initial cell state
   T* initial_cell_state = nullptr;
   if (initial_c.GetShape().size() > 0) {
      if (fLayout == 0) {
         initial_cell_state = initial_c.GetData();
      } else {
         initial_cell_state = new T[num_directions * batch_size * fHiddenSize];
         for (size_t direction = 0; direction < num_directions; direction++) {
            for (size_t batch = 0; batch < batch_size; batch++) {
               for (size_t h = 0; h < fHiddenSize; h++) {
                  initial_cell_state[direction * batch_size * fHiddenSize + batch * fHiddenSize + h] =
                     initial_c.GetData()[batch * num_directions * fHiddenSize + direction * fHiddenSize
                        + h];
               }
            }
         }
      }
   }

   // Set the feedforward
   T ff_input_gate[seq_length * batch_size * fHiddenSize];
   T* ff_forget_gate = nullptr;
   if (fInputForget == 0) {
      ff_forget_gate = new T[seq_length * batch_size * fHiddenSize];
   }
   T ff_output_gate[seq_length * batch_size * fHiddenSize];
   T ff_cell_gate[seq_length * batch_size * fHiddenSize];
   // Set the gates
   size_t hidden_state_size = seq_length * num_directions * batch_size * fHiddenSize;
   T input_gate[hidden_state_size];
   T* forget_gate = nullptr;
   if (fInputForget == 0) {
      forget_gate = new T[hidden_state_size];
   }
   T output_gate[hidden_state_size];
   T cell_gate[hidden_state_size];
   // Set the cell state
   T cell_state[hidden_state_size];
   // new cell state = h(cell_state)
   T new_cell_state[hidden_state_size];
   // Set the hidden state
   T * hidden_state = nullptr;
   if (fLayout == 0 && Y.GetShape().size() > 0) {
      hidden_state = Y.GetData();
   } else {
      hidden_state = new T[hidden_state_size];
   }

   for (size_t direction = 0; direction < num_directions; direction++) {
      char transA = 'N';
      char transB = 'T';
      int m = seq_length * batch_size;
      int n = fHiddenSize;
      int k = input_size;
      float alpha = 1.;
      float beta = 0.;
      // input_gate = input * weight^T + bias
      size_t wi_offset = direction * 4 * fHiddenSize * input_size;
      BLAS::sgemm_(&transB, &transA, &n, &m, &k, &alpha, W.GetData() + wi_offset, &k,
         input, &k, &beta, ff_input_gate, &n);
      // output_gate = input * weight^T + bias
      size_t wo_offset = direction * 4 * fHiddenSize * input_size +
         2 * fHiddenSize * input_size;
      BLAS::sgemm_(&transB, &transA, &n, &m, &k, &alpha, W.GetData() + wo_offset, &k,
         input, &k, &beta, ff_output_gate, &n);
      // cell_gate = input * weight^T + bias
      size_t wc_offset = direction * 4 * fHiddenSize * input_size +
         3 * fHiddenSize * input_size;
      BLAS::sgemm_(&transB, &transA, &n, &m, &k, &alpha, W.GetData() + wc_offset, &k,
         input, &k, &beta, ff_cell_gate, &n);
      // forget_gate = input * weight^T + bias
      if (fInputForget == 0) {
         size_t wf_offset = direction * 4 * fHiddenSize * input_size +
            fHiddenSize * input_size;
         BLAS::sgemm_(&transB, &transA, &n, &m, &k, &alpha, W.GetData() + wf_offset, &k,
            input, &k, &beta, ff_forget_gate, &n);
      }
      if (bias) {
         int bias_size = seq_length * batch_size * fHiddenSize;
         int incx = 1;
         int incy = 1;
         // ff_input_gate += bias_i
         size_t bi_offset = direction * seq_length * batch_size * fHiddenSize;
         BLAS::saxpy_(&bias_size, &alpha, bias + bi_offset, &incx, ff_input_gate, &incy);
         // ff_output_gate += bias_o
         size_t bo_offset = 2 * num_directions * seq_length * batch_size * fHiddenSize
            + direction * seq_length * batch_size * fHiddenSize;
         BLAS::saxpy_(&bias_size, &alpha, bias + bo_offset, &incx, ff_output_gate, &incy);
         // ff_cell_gate += bias_c
         size_t bc_offset = 3 * num_directions * seq_length * batch_size * fHiddenSize
            + direction * seq_length * batch_size * fHiddenSize;
         BLAS::saxpy_(&bias_size, &alpha, bias + bc_offset, &incx, ff_cell_gate, &incy);
         if (fInputForget == 0) {
            // ff_forget_gate += bias_f
            size_t bf_offset = num_directions * seq_length * batch_size * fHiddenSize
               + direction * seq_length * batch_size * fHiddenSize;
            BLAS::saxpy_(&bias_size, &alpha, bias + bf_offset, &incx, ff_forget_gate, &incy);
         }
      }
      // copy ff_input_gate, ff_output_gate and ff_cell_gate and ff_forget_gate into input_gate, output_gate,
      //   cell_gate and forget_gate
      for (size_t seq = 0; seq < seq_length; seq++) {
         size_t ff_offset = seq * batch_size * fHiddenSize;
         size_t ff_size = batch_size * fHiddenSize;
         size_t gate_offset = seq * num_directions * batch_size * fHiddenSize +
            direction * batch_size * fHiddenSize;
         std::copy(ff_input_gate + ff_offset, ff_input_gate + ff_offset + ff_size,
            input_gate + gate_offset);
         std::copy(ff_output_gate + ff_offset, ff_output_gate + ff_offset + ff_size,
            output_gate + gate_offset);
         std::copy(ff_cell_gate + ff_offset, ff_cell_gate + ff_offset + ff_size,
            cell_gate + gate_offset);
         if (fInputForget == 0) {
            std::copy(ff_forget_gate + ff_offset, ff_forget_gate + ff_offset + ff_size,
               forget_gate + gate_offset);
         }
      }

      bool backward = (fDirection == "backward") || (direction == 1);
      for (size_t seq = 0; seq < seq_length; seq++) {
         size_t index = backward ? seq_length - 1 - seq : seq;
         int m2 = batch_size;
         size_t offset = index * num_directions * batch_size * fHiddenSize
            + direction * batch_size * fHiddenSize;
         size_t size = batch_size * fHiddenSize;
         // gate = 1.0 * gate + previous_hidden_state * R^T
         if (seq == 0) {
            if (initial_hidden_state) {
               size_t initial_h_offset = direction * batch_size * fHiddenSize;
               size_t ri_offset = direction * 4 * fHiddenSize * fHiddenSize;
               BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + ri_offset, &n,
                  initial_hidden_state + initial_h_offset, &n, &alpha, input_gate + offset, &n);
               size_t ro_offset = direction * 4 * fHiddenSize * fHiddenSize +
                  2 * fHiddenSize * fHiddenSize;
               BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + ro_offset, &n,
                  initial_hidden_state + initial_h_offset, &n, &alpha, output_gate + offset, &n);
               size_t rc_offset = direction * 4 * fHiddenSize * fHiddenSize +
                  3 * fHiddenSize * fHiddenSize;
               BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + rc_offset, &n,
                  initial_hidden_state + initial_h_offset, &n, &alpha, cell_gate + offset, &n);
               if (fInputForget == 0) {
                  size_t rf_offset = direction * 4 * fHiddenSize * fHiddenSize +
                     fHiddenSize * fHiddenSize;
                  BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + rf_offset, &n,
                     initial_hidden_state + initial_h_offset, &n, &alpha, forget_gate + offset, &n);
               }
            }
         } else {
            size_t previous_offset = (backward ? (index + 1) : (seq - 1)) * num_directions * batch_size *
               fHiddenSize + direction * batch_size * fHiddenSize;
            size_t ri_offset = direction * 4 * fHiddenSize * fHiddenSize;
            BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + ri_offset, &n,
               hidden_state + previous_offset, &n, &alpha, input_gate + offset, &n);
            size_t ro_offset = direction * 4 * fHiddenSize * fHiddenSize +
               2 * fHiddenSize * fHiddenSize;
            BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + ro_offset, &n,
               hidden_state + previous_offset, &n, &alpha, output_gate + offset, &n);
            size_t rc_offset = direction * 4 * fHiddenSize * fHiddenSize +
               3 * fHiddenSize * fHiddenSize;
            BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + rc_offset, &n,
               hidden_state + previous_offset, &n, &alpha, cell_gate + offset, &n);
            if (fInputForget == 0) {
               size_t rf_offset = direction * 4 * fHiddenSize * fHiddenSize +
                  fHiddenSize * fHiddenSize;
               BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + rf_offset, &n,
                  hidden_state + previous_offset, &n, &alpha, forget_gate + offset, &n);
            }
         }
         // Clip the elements of the cell gate into the range [-fClip, fClip]
         if (fClip > 0.) {
            for (size_t i = offset; i < offset + size; i++) {
               T x = (cell_gate[i] > -fClip) ? cell_gate[i] : -fClip;
               cell_gate[i] = (x < fClip)? x : fClip;
            }
         }
         // Apply the activation function to the cell gate, cell_gate = g(cell_gate)
         if (fActivations[direction * 3 + 1] == "Relu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (cell_gate[i] < 0.)
                  cell_gate[i] = 0.;
            }
         } else if (fActivations[direction * 3 + 1] == "Tanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float ex = exp(-2 * cell_gate[i]);
               cell_gate[i] = (1. - ex) / (1. + ex);
            }
         } else if (fActivations[direction * 3 + 1] == "Sigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               cell_gate[i] = 1. / (1. + exp(-cell_gate[i]));
            }
         } else if (fActivations[direction * 3 + 1] == "Affine") {
            for (size_t i = offset; i < offset + size; i++) {
               cell_gate[i] = fActivationAlpha[direction * 3 + 1] * cell_gate[i]
                  + fActivationBeta[direction * 3 + 1];
            }
         } else if (fActivations[direction * 3 + 1] == "LeakyRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (cell_gate[i] < 0.) {
                  cell_gate[i] = fActivationAlpha[direction * 3 + 1] * cell_gate[i];
               }
            }
         } else if (fActivations[direction * 3 + 1] == "ThresholdRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (cell_gate[i] < fActivationAlpha[direction * 3 + 1]) {
                  cell_gate[i] = 0.;
               }
            }
         } else if (fActivations[direction * 3 + 1] == "ScaledTanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float x = exp(-2 * fActivationBeta[direction * 3 + 1] * cell_gate[i]);
               cell_gate[i] = fActivationAlpha[direction * 3 + 1] * (1. - x) / (1. + x);
            }
         } else if (fActivations[direction * 3 + 1] == "HardSigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               float a = fActivationAlpha[direction * 3 + 1] * cell_gate[i] + fActivationBeta[direction * 3 + 1];
               float b = (a > 0.) ? a : 0.;
               cell_gate[i] = (b < 1.) ? b : 1.;
            }
         } else if (fActivations[direction * 3 + 1] == "Elu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (cell_gate[i] < 0.) {
                  cell_gate[i] = fActivationAlpha[direction * 3 + 1] * (exp(cell_gate[i] - 1.));
               }
            }
         } else if (fActivations[direction * 3 + 1] == "Softsign") {
            for (size_t i = offset; i < offset + size; i++) {
               cell_gate[i] = cell_gate[i] / (1. + abs(new_cell_state[i]));
            }
         } else { // Softplus
            for (size_t i = offset; i < offset + size; i++) {
               cell_gate[i] = log(1. + exp(cell_gate[i]));
            }
         }
         // Peephole connections for the input gate and the forget gate
         if (peephole) {
            // gate = 1.0 * gate + previous_cell_state * P^T
            if (seq == 0) {
               if (initial_cell_state) {
                  size_t pi_offset = direction * 3 * batch_size * fHiddenSize;
                  size_t initial_c_offset = direction * batch_size * fHiddenSize;
                  for (size_t i = 0; i < size; i++) {
                     input_gate[i + offset] += peephole[i + pi_offset] * initial_cell_state[i + initial_c_offset];
                  }
                  if (fInputForget == 0) {
                     size_t pf_offset = direction * 3 * batch_size * fHiddenSize +
                        batch_size * fHiddenSize;
                     for (size_t i = 0; i < size; i++) {
                        forget_gate[i + offset] += peephole[i + pf_offset] * initial_cell_state[i + initial_c_offset];
                     }
                  }
               }
            } else {
               size_t pi_offset = direction * 3 * batch_size * fHiddenSize;
               size_t c_offset = (backward ? (index + 1) : (seq - 1)) * num_directions * batch_size *
                  fHiddenSize + direction * batch_size * fHiddenSize;
               for (size_t i = 0; i < size; i++) {
                  input_gate[i + offset] += peephole[i + pi_offset] * cell_state[i + c_offset];
               }
               if (fInputForget == 0) {
                  size_t pf_offset = direction * 3 * batch_size * fHiddenSize +
                     batch_size * fHiddenSize;
                  for (size_t i = 0; i < size; i++) {
                     forget_gate[i + offset] += peephole[i + pf_offset] * cell_state[i + c_offset];
                  }
               }
            }
         }
         // Clip the elements of the input gate into the range [-fClip, fClip]
         if (fClip > 0.) {
            for (size_t i = offset; i < offset + size; i++) {
               T x = (input_gate[i] > -fClip) ? input_gate[i] : -fClip;
               input_gate[i] = (x < fClip)? x : fClip;
            }
         }
         // Apply the activation function to the input gate
         if (fActivations[direction * 3] == "Relu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (input_gate[i] < 0.)
                  input_gate[i] = 0.;
            }
         } else if (fActivations[direction * 3] == "Tanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float ex = exp(-2 * input_gate[i]);
               input_gate[i] = (1. - ex) / (1. + ex);
            }
         } else if (fActivations[direction * 3] == "Sigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               input_gate[i] = 1. / (1. + exp(-input_gate[i]));
            }
         } else if (fActivations[direction * 3] == "Affine") {
            for (size_t i = offset; i < offset + size; i++) {
               input_gate[i] = fActivationAlpha[direction * 3] * input_gate[i]
                  + fActivationBeta[direction * 3];
            }
         } else if (fActivations[direction * 3] == "LeakyRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (input_gate[i] < 0.) {
                  input_gate[i] = fActivationAlpha[direction * 3] * input_gate[i];
               }
            }
         } else if (fActivations[direction * 3] == "ThresholdRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (input_gate[i] < fActivationAlpha[direction * 3]) {
                  input_gate[i] = 0.;
               }
            }
         } else if (fActivations[direction * 3] == "ScaledTanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float x = exp(-2 * fActivationBeta[direction * 3] * input_gate[i]);
               input_gate[i] = fActivationAlpha[direction * 3] * (1. - x) / (1. + x);
            }
         } else if (fActivations[direction * 3] == "HardSigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               float a = fActivationAlpha[direction * 3] * input_gate[i] + fActivationBeta[direction * 3];
               float b = (a > 0.) ? a : 0.;
               input_gate[i] = (b < 1.) ? b : 1.;
            }
         } else if (fActivations[direction * 3] == "Elu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (input_gate[i] < 0.) {
                  input_gate[i] = fActivationAlpha[direction * 3] * (exp(input_gate[i] - 1.));
               }
            }
         } else if (fActivations[direction * 3] == "Softsign") {
            for (size_t i = offset; i < offset + size; i++) {
               input_gate[i] = input_gate[i] / (1. + abs(new_cell_state[i]));
            }
         } else { // Softplus
            for (size_t i = offset; i < offset + size; i++) {
               input_gate[i] = log(1. + exp(input_gate[i]));
            }
         }

         if (fInputForget == 0) {
            // Clip the elements of the forget gate into the range [-fClip, fClip]
            if (fClip > 0.) {
               for (size_t i = offset; i < offset + size; i++) {
                  T x = (forget_gate[i] > -fClip) ? forget_gate[i] : -fClip;
                  forget_gate[i] = (x < fClip)? x : fClip;
               }
            }
            // Apply the activation function to the forget gate
            if (fActivations[direction * 3] == "Relu") {
               for (size_t i = offset; i < offset + size; i++) {
                  if (forget_gate[i] < 0.)
                     forget_gate[i] = 0.;
               }
            } else if (fActivations[direction * 3] == "Tanh") {
               for (size_t i = offset; i < offset + size; i++) {
                  float ex = exp(-2 * forget_gate[i]);
                  forget_gate[i] = (1. - ex) / (1. + ex);
               }
            } else if (fActivations[direction * 3] == "Sigmoid") {
               for (size_t i = offset; i < offset + size; i++) {
                  forget_gate[i] = 1. / (1. + exp(-forget_gate[i]));
               }
            } else if (fActivations[direction * 3] == "Affine") {
               for (size_t i = offset; i < offset + size; i++) {
                  forget_gate[i] = fActivationAlpha[direction * 3] * forget_gate[i]
                     + fActivationBeta[direction * 3];
               }
            } else if (fActivations[direction * 3] == "LeakyRelu") {
               for (size_t i = offset; i < offset + size; i++) {
                  if (forget_gate[i] < 0.) {
                     forget_gate[i] = fActivationAlpha[direction * 3] * forget_gate[i];
                  }
               }
            } else if (fActivations[direction * 3] == "ThresholdRelu") {
               for (size_t i = offset; i < offset + size; i++) {
                  if (forget_gate[i] < fActivationAlpha[direction * 3]) {
                     forget_gate[i] = 0.;
                  }
               }
            } else if (fActivations[direction * 3] == "ScaledTanh") {
               for (size_t i = offset; i < offset + size; i++) {
                  float x = exp(-2 * fActivationBeta[direction * 3] * forget_gate[i]);
                  forget_gate[i] = fActivationAlpha[direction * 3] * (1. - x) / (1. + x);
               }
            } else if (fActivations[direction * 3] == "HardSigmoid") {
               for (size_t i = offset; i < offset + size; i++) {
                  float a = fActivationAlpha[direction * 3] * forget_gate[i] + fActivationBeta[direction * 3];
                  float b = (a > 0.) ? a : 0.;
                  forget_gate[i] = (b < 1.) ? b : 1.;
               }
            } else if (fActivations[direction * 3] == "Elu") {
               for (size_t i = offset; i < offset + size; i++) {
                  if (forget_gate[i] < 0.) {
                     forget_gate[i] = fActivationAlpha[direction * 3] * (exp(forget_gate[i] - 1.));
                  }
               }
            } else if (fActivations[direction * 3] == "Softsign") {
               for (size_t i = offset; i < offset + size; i++) {
                  forget_gate[i] = forget_gate[i] / (1. + abs(new_cell_state[i]));
               }
            } else { // Softplus
               for (size_t i = offset; i < offset + size; i++) {
                  forget_gate[i] = log(1. + exp(forget_gate[i]));
               }
            }
         }
         // cell_state = input_gate o cell_gate
         for (size_t i = offset; i < offset + size; i++) {
            cell_state[i] = input_gate[i] * cell_gate[i];
         }
         if (fInputForget == 0) {
            if (seq == 0) {
               if (initial_cell_state) {
                  // cell_state += forget_gate o initial_cell_state
                  for (size_t i = 0; i < size; i++) {
                     cell_state[i + offset] += forget_gate[i + offset] * initial_cell_state[i];
                  }
               }
            } else {
               // cell_state += forget_gate o previous_cell_state
               size_t previous_offset = (backward ? (index + 1) : (seq - 1)) * num_directions * batch_size *
                  fHiddenSize + direction * batch_size * fHiddenSize;
               for (size_t i = 0; i < size; i++) {
                  cell_state[i + offset] += forget_gate[i + offset] * cell_state[i + previous_offset];
               }
            }
         }
         if (peephole) {
            // Peephole connection for the output gate
            size_t p_offset = direction * 3 * batch_size * fHiddenSize +
               2 * batch_size * fHiddenSize;
            for (size_t i = 0; i < size; i++) {
               output_gate[i + offset] += peephole[i + p_offset] * cell_state[i + offset];
            }
         }
         // Clip the elements of the output gate into the range [-fClip, fClip]
         if (fClip > 0.) {
            for (size_t i = offset; i < offset + size; i++) {
               T x = (output_gate[i] > -fClip) ? output_gate[i] : -fClip;
               output_gate[i] = (x < fClip)? x : fClip;
            }
         }
         // Apply the activation function to the output gate, output_gate = f(output_gate)
         if (fActivations[direction * 3] == "Relu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (output_gate[i] < 0.)
                  output_gate[i] = 0.;
            }
         } else if (fActivations[direction * 3] == "Tanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float ex = exp(-2 * output_gate[i]);
               output_gate[i] = (1. - ex) / (1. + ex);
            }
         } else if (fActivations[direction * 3] == "Sigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               output_gate[i] = 1. / (1. + exp(-output_gate[i]));
            }
         } else if (fActivations[direction * 3] == "Affine") {
            for (size_t i = offset; i < offset + size; i++) {
               output_gate[i] = fActivationAlpha[direction * 3] * output_gate[i]
                  + fActivationBeta[direction * 3];
            }
         } else if (fActivations[direction * 3] == "LeakyRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (output_gate[i] < 0.) {
                  output_gate[i] = fActivationAlpha[direction * 3] * output_gate[i];
               }
            }
         } else if (fActivations[direction * 3] == "ThresholdRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (output_gate[i] < fActivationAlpha[direction * 3]) {
                  output_gate[i] = 0.;
               }
            }
         } else if (fActivations[direction * 3] == "ScaledTanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float x = exp(-2 * fActivationBeta[direction * 3] * output_gate[i]);
               output_gate[i] = fActivationAlpha[direction * 3] * (1. - x) / (1. + x);
            }
         } else if (fActivations[direction * 3] == "HardSigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               float a = fActivationAlpha[direction * 3] * output_gate[i] + fActivationBeta[direction * 3];
               float b = (a > 0.) ? a : 0.;
               output_gate[i] = (b < 1.) ? b : 1.;
            }
         } else if (fActivations[direction * 3] == "Elu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (output_gate[i] < 0.) {
                  output_gate[i] = fActivationAlpha[direction * 3] * (exp(output_gate[i] - 1.));
               }
            }
         } else if (fActivations[direction * 3] == "Softsign") {
            for (size_t i = offset; i < offset + size; i++) {
               output_gate[i] = output_gate[i] / (1. + abs(new_cell_state[i]));
            }
         } else { // Softplus
            for (size_t i = offset; i < offset + size; i++) {
               output_gate[i] = log(1. + exp(output_gate[i]));
            }
         }
         // copy cell_state into new_cell_state
         std::copy(cell_state + offset, cell_state + offset + size, new_cell_state + offset);
         // Clip the elements of new_cell_state into the range [-fClip, fClip]
         if (fClip > 0.) {
            for (size_t i = offset; i < offset + size; i++) {
               T x = (new_cell_state[i] > -fClip) ? new_cell_state[i] : -fClip;
               new_cell_state[i] = (x < fClip) ? x : fClip;
            }
         }
         // new_cell_state = h(new_cell_state)
         if (fActivations[direction * 3 + 2] == "Relu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (new_cell_state[i] < 0.)
                  new_cell_state[i] = 0.;
            }
         } else if (fActivations[direction * 3 + 2] == "Tanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float ex = exp(-2 * new_cell_state[i]);
               new_cell_state[i] = (1. - ex) / (1. + ex);
            }
         } else if (fActivations[direction * 3 + 2] == "Sigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               new_cell_state[i] = 1. / (1. + exp(-new_cell_state[i]));
            }
         } else if (fActivations[direction * 3 + 2] == "Affine") {
            for (size_t i = offset; i < offset + size; i++) {
               new_cell_state[i] = fActivationAlpha[direction * 3 + 2] * new_cell_state[i]
                  + fActivationBeta[direction * 3 + 2];
            }
         } else if (fActivations[direction * 3 + 2] == "LeakyRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (new_cell_state[i] < 0.) {
                  new_cell_state[i] = fActivationAlpha[direction * 3 + 2] * new_cell_state[i];
               }
            }
         } else if (fActivations[direction * 3 + 2] == "ThresholdRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (new_cell_state[i] < fActivationAlpha[direction * 3 + 2]) {
                  new_cell_state[i] = 0.;
               }
            }
         } else if (fActivations[direction * 3 + 2] == "ScaledTanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float x = exp(-2 * fActivationBeta[direction * 3 + 2] * new_cell_state[i]);
               new_cell_state[i] = fActivationAlpha[direction * 3 + 2] * (1. - x) / (1. + x);
            }
         } else if (fActivations[direction * 3 + 2] == "HardSigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               float a = fActivationAlpha[direction * 3 + 2] * new_cell_state[i] + fActivationBeta[direction * 3 + 2];
               float b = (a > 0.) ? a : 0.;
               new_cell_state[i] = (b < 1.) ? b : 1.;
            }
         } else if (fActivations[direction * 3 + 2] == "Elu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (new_cell_state[i] < 0.) {
                  new_cell_state[i] = fActivationAlpha[direction * 3 + 2] * (exp(new_cell_state[i] - 1.));
               }
            }
         } else if (fActivations[direction * 3 + 2] == "Softsign") {
            for (size_t i = offset; i < offset + size; i++) {
               new_cell_state[i] = new_cell_state[i] / (1. + abs(new_cell_state[i]));
            }
         } else { // Softplus
            for (size_t i = offset; i < offset + size; i++) {
               new_cell_state[i] = log(1. + exp(new_cell_state[i]));
            }
         }
         // hidden_state = output_gate o new_cell_state
         for (size_t i = offset; i < offset + size; i++) {
            hidden_state[i] = output_gate[i] * new_cell_state[i];
         }
      }
   }

   // LSTM with different sequence lengths
   if (sequence_lens.GetShape().size() > 0) {
      for (size_t seq = 0; seq < seq_length; seq++) {
         for (size_t batch = 0; batch < batch_size; batch++) {
            if (seq >= sequence_lens.GetData()[batch]) {
               for (size_t direction = 0; direction < num_directions; direction++) {
                  for (size_t h = 0; h < fHiddenSize; h++) {
                     size_t idx = seq * num_directions * batch_size * fHiddenSize
                        + direction * batch_size * fHiddenSize + batch * fHiddenSize + h;
                     cell_state[idx] = 0.;
                     hidden_state[idx] = 0.;
                  }
               }
            }
         }
      }
   }

   // copy hidden_state into Y and Y_h, and copy cell_state into Y_c
   if (fLayout == 0) {
      if (Y_h.GetShape().size() > 0) {
         if (sequence_lens.GetShape().size() > 0) {
            for (size_t direction = 0; direction < num_directions; direction++) {
               bool backward = (fDirection == "backward") || (direction == 1);
               for (size_t batch = 0; batch < batch_size; batch++) {
                  size_t seq = backward ? 0 : (sequence_lens.GetShape().size() > 0 ?
                     sequence_lens.GetData()[batch] - 1 : seq_length - 1);
                  size_t offset = seq * num_directions * batch_size * fHiddenSize
                     + direction * batch_size * fHiddenSize + batch * fHiddenSize;
                  size_t y_h_offset = direction * batch_size * fHiddenSize
                     + batch * fHiddenSize;
                  std::copy(hidden_state + offset, hidden_state + offset + fHiddenSize,
                     Y_h.GetData() + y_h_offset);
               }
            }
         } else {
            for (size_t direction = 0; direction < num_directions; direction++) {
               bool backward = (fDirection == "backward") || (direction == 1);
               size_t seq = backward ? 0 : seq_length - 1;
               size_t offset = seq * num_directions * batch_size * fHiddenSize +
                  direction * batch_size * fHiddenSize;
               size_t size = batch_size * fHiddenSize;
               size_t y_h_offset = direction * batch_size * fHiddenSize;
               std::copy(hidden_state + offset, hidden_state + offset + size,
                  Y_h.GetData() + y_h_offset);
            }
         }
      }
      if (Y_c.GetShape().size() > 0) {
         if (sequence_lens.GetShape().size() > 0) {
            for (size_t direction = 0; direction < num_directions; direction++) {
               bool backward = (fDirection == "backward") || (direction == 1);
               for (size_t batch = 0; batch < batch_size; batch++) {
                  size_t seq = backward ? 0 : (sequence_lens.GetShape().size() > 0 ?
                     sequence_lens.GetData()[batch] - 1 : seq_length - 1);
                  size_t offset = seq * num_directions * batch_size * fHiddenSize
                     + direction * batch_size * fHiddenSize + batch * fHiddenSize;
                  size_t y_c_offset = direction * batch_size * fHiddenSize
                     + batch * fHiddenSize;
                  std::copy(cell_state + offset, hidden_state + offset + fHiddenSize,
                     Y_c.GetData() + y_c_offset);
               }
            }
         } else {
            for (size_t direction = 0; direction < num_directions; direction++) {
               bool backward = (fDirection == "backward") || (direction == 1);
               size_t seq = backward ? 0 : seq_length - 1;
               size_t offset = seq * num_directions * batch_size * fHiddenSize +
                  direction * batch_size * fHiddenSize;
               size_t size = batch_size * fHiddenSize;
               size_t y_c_offset = direction * batch_size * fHiddenSize;
               std::copy(cell_state + offset, cell_state + offset + size,
                  Y_c.GetData() + y_c_offset);
            }
         }
      }
   } else { // fLayout=1
      if (Y.GetShape().size() > 0) {
         for (size_t seq = 0; seq < seq_length; seq++) {
            for (size_t direction = 0; direction < num_directions; direction++) {
               for (size_t batch = 0; batch < batch_size; batch++) {
                  size_t offset = seq * num_directions * batch_size * fHiddenSize +
                     direction * batch_size * fHiddenSize + batch * fHiddenSize;
                  size_t y_offset = batch * seq_length * num_directions * fHiddenSize +
                     seq * num_directions * fHiddenSize + direction * fHiddenSize;
                  std::copy(hidden_state + offset, hidden_state + offset + fHiddenSize,
                            Y.GetData() + y_offset);
               }
            }
         }
      }
      if (Y_h.GetShape().size() > 0) {
         for (size_t direction = 0; direction < num_directions; direction++) {
            bool backward = (fDirection == "backward") || (direction == 1);
            for (size_t batch = 0; batch < batch_size; batch++) {
               size_t seq = backward ? 0 : (sequence_lens.GetShape().size() > 0 ?
                  sequence_lens.GetData()[batch] - 1 : seq_length - 1);
               size_t offset = seq * num_directions * batch_size * fHiddenSize +
                  direction * batch_size * fHiddenSize + batch * fHiddenSize;
               size_t y_h_offset = batch * num_directions * fHiddenSize +
                  direction * fHiddenSize;
               std::copy(hidden_state + offset, hidden_state + offset + fHiddenSize,
                  Y_h.GetData() + y_h_offset);
            }
         }
      }
      if (Y_c.GetShape().size() > 0) {
         for (size_t direction = 0; direction < num_directions; direction++) {
            bool backward = (fDirection == "backward") || (direction == 1);
            for (size_t batch = 0; batch < batch_size; batch++) {
               size_t seq = backward ? 0 : (sequence_lens.GetShape().size() > 0 ?
                  sequence_lens.GetData()[batch] - 1 : seq_length - 1);
               size_t offset = seq * num_directions * batch_size * fHiddenSize +
                  direction * batch_size * fHiddenSize + batch * fHiddenSize;
               size_t y_c_offset = batch * num_directions * fHiddenSize +
                  direction * fHiddenSize;
               std::copy(cell_state + offset, cell_state + offset + fHiddenSize,
                  Y_c.GetData() + y_c_offset);
            }
         }
      }
   }

   if (bias)
      delete[] bias;

   if (fInputForget == 0) {
      delete[] ff_forget_gate;
      delete[] forget_gate;
   }

   if (fLayout == 1) {
      delete[] input;
      delete[] initial_hidden_state;
      delete[] initial_cell_state;
      if (Y.GetShape().size() == 0)
         delete[] hidden_state;
   }
   if (peephole && batch_size > 1)
      delete[] peephole;
}


}
}
}

#endif
