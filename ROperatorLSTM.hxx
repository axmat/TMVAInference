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
   if (fHiddenSize != W.GetShape()[1]) {
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
            // Compute the bias_xh + bias_hh
            for (size_t h = 0; h < fHiddenSize; h++) {
               sum[h] = B.GetData()[direction * 8 * fHiddenSize + gate * fHiddenSize + h] +
                  B.GetData()[direction * 8 * fHiddenSize + gate * fHiddenSize + h + 4*fHiddenSize];
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
   T feedforward[seq_length * batch_size * fHiddenSize];
   // Set the gates
   size_t hidden_state_size = seq_length * num_directions * batch_size * fHiddenSize;
   T input_gate[hidden_state_size];
   T* forget_gate = nullptr;
   if (fInputForget == 0) {
      forget_gate = new T[hidden_state_size];
   }
   T output_gate[hidden_state_size];
   T cell_gate[hidden_state_size];
   T* gates[4] = {input_gate, forget_gate, output_gate, cell_gate};
   // Set the cell state
   T cell_state[hidden_state_size];
   T new_cell_state[hidden_state_size];
   // Set the hidden state
   T * hidden_state = nullptr;
   if (fLayout == 0 && Y.GetShape().size() > 0) {
      hidden_state = Y.GetData();
   } else {
      hidden_state = new T[hidden_state_size];
   }

   for (size_t direction = 0; direction < num_directions; direction++) {
      for (size_t gate = 0; gate < 4; gate++) {
         if (gate == 1 && fInputForget == 1)
            continue;
         // feedforward = input * weight^T + bias
         char transA = 'N';
         char transB = 'T';
         int m = seq_length * batch_size;
         int n = fHiddenSize;
         int k = input_size;
         float alpha = 1.;
         float beta = 0.;
         size_t w_offset = direction * 4 * fHiddenSize * input_size +
            gate * fHiddenSize * input_size;
         BLAS::sgemm_(&transB, &transA, &n, &m, &k, &alpha, W.GetData() + w_offset, &k,
            input, &k, &beta, feedforward, &n);
         if (bias) {
            int bias_size = seq_length * batch_size * fHiddenSize;
            int incx = 1;
            int incy = 1;
            size_t bias_offset = gate * num_directions * seq_length * batch_size * fHiddenSize
               + direction * seq_length * batch_size * fHiddenSize;
            BLAS::saxpy_(&bias_size, &alpha, bias + bias_offset, &incx, feedforward, &incy);
         }
         // Copy feedforward into the gate
         for (size_t seq = 0; seq < seq_length; seq++) {
            size_t feedforward_offset = seq * batch_size * fHiddenSize;
            size_t feedforward_size = batch_size * fHiddenSize;
            size_t offset = seq * num_directions * batch_size * fHiddenSize +
                         direction * batch_size * fHiddenSize;
            std::copy(feedforward + feedforward_offset, feedforward + feedforward_offset + feedforward_size,
                   gates[gate] + offset);
         }
      }

      bool backward = (fDirection == "backward") || (direction == 1);
      for (size_t seq = 0; seq < seq_length; seq++) {
         size_t index = backward ? seq_length - 1 - seq : seq;
         char transA = 'N';
         char transB = 'N';
         int m = batch_size;
         int n = fHiddenSize;
         float alpha = 1.;
         size_t offset = index * num_directions * batch_size * fHiddenSize
            + direction * batch_size * fHiddenSize;
         size_t size = batch_size * fHiddenSize;
         for (size_t gate = 0; gate < 4; gate++) {
            if (gate == 1 && fInputForget == 1)
               continue;
            // gate = 1.0 * gate + previous_hidden_state * R^T
            if (seq == 0) {
               if (initial_hidden_state) {
                  size_t r_offset = direction * 4 * fHiddenSize * fHiddenSize +
                     gate * fHiddenSize * fHiddenSize;
                  size_t initial_h_offset = direction * batch_size * fHiddenSize;
                  BLAS::sgemm_(&transB, &transA, &n, &m, &n, &alpha, R.GetData() + r_offset, &n,
                     initial_hidden_state + initial_h_offset, &n, &alpha,
                     gates[gate] + offset, &n);
               }
            } else {
               size_t r_offset = direction * 4 * fHiddenSize * fHiddenSize +
                  gate * fHiddenSize * fHiddenSize;
               size_t previous_offset = (backward ? (index + 1) : (seq - 1)) * num_directions * batch_size *
                  fHiddenSize + direction * batch_size * fHiddenSize;
               BLAS::sgemm_(&transB, &transA, &n, &m, &n, &alpha, R.GetData() + r_offset, &n,
                  gates[gate] + previous_offset, &n, &alpha, gates[gate] + offset, &n);
            }
         }
         // TODO Peepholes LSTM
         for (size_t gate = 0; gate < 3; gate++) {
            // Clip the elements of the input gate, the forget gate and the output gate into the range [-fClip, fClip]
            if (fClip > 0.) {
               for (size_t i = offset; i < offset + size; i++) {
                  T x = (gates[gate][i] > -fClip) ? gates[gate][i] : -fClip;
                  gates[gate][i] = (x < fClip)? x : fClip;
               }
            }
            // Apply the activation function to the input gate, the forget gate and the output gate
            if (fActivations[direction * 3] == "Relu") {
               for (size_t i = offset; i < offset + size; i++) {
                  if (gates[gate][i] < 0.)
                     gates[gate][i] = 0.;
               }
            } else if (fActivations[direction * 3] == "Tanh") {
               for (size_t i = offset; i < offset + size; i++) {
                  float ex = exp(-2 * gates[gate][i]);
                  gates[gate][i] = (1. - ex) / (1. + ex);
               }
            } else if (fActivations[direction * 3] == "Sigmoid") {
               for (size_t i = offset; i < offset + size; i++) {
                  gates[gate][i] = 1. / (1. + exp(-gates[gate][i]));
               }
            } else {
               throw std::runtime_error("TMVA - Activation function " + fActivations[direction * 3] +
                  " not implemented.");
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
         } else {
            throw std::runtime_error("TMVA - Activation function " + fActivations[direction * 3 + 1] +
               " not implemented.");
         }
         // cell_state = input_gate o cell_gate
         for (size_t i = offset; i < offset + size; i++) {
            cell_state[i] = input_gate[i] * cell_gate[i];
         }
         if (fInputForget == 0) {
            if (seq == 0) {
               // cell_state += forget_gate o initial_cell_state
               for (size_t i = offset; i < offset + size; i++) {
                  cell_state[i] += forget_gate[i] * initial_cell_state[i];
               }
            } else {
               // cell_state += forget_gate o previous_cell_state
               size_t previous_offset;
               for (size_t i = 0; i < size; i++) {
                  cell_state[i + offset] += forget_gate[i + offset] * cell_state[i + previous_offset];
               }
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
         } else {
            throw std::runtime_error("TMVA - Activation function " + fActivations[direction * 3 + 2] +
               " not implemented.");
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
               std::copy(cell_state + offset, hidden_state + offset + size,
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
               std::copy(cell_state + offset, hidden_state + offset + fHiddenSize,
                  Y_c.GetData() + y_c_offset);
            }
         }
      }
   }

   if (bias)
      delete[] bias;

   if (fLayout == 1) {
      delete[] input;
      delete[] initial_hidden_state;
      delete[] initial_cell_state;
      if (Y.GetShape().size() == 0)
         delete[] hidden_state;
   }
}


}
}
}

#endif
