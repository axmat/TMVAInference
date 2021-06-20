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
                     RTensor<T> &sequence_lens,
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
                                    RTensor<T> &sequence_lens,
                                    RTensor<T> &initial_h,
                                    RTensor<T> &initial_c,
                                    RTensor<T> &P,
                                    RTensor<T> &Y,
                                    RTensor<T> &Y_h,
                                    RTensor<T> &Y_c) {
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

   // Activation functions
   if (fActivations.empty()) {
      if (fDirection == "forward" || fDirection == "backward") {
         fActivations = {"Tanh"};
      } else {
         fActivations = {"Tanh", "Tanh"};
      }
   }
   // Default values of alpha and beta
   for (size_t i = 0; i < fActivations.size(); i++) {
      if (fActivations[i] == "LeakyRelu") {
         if (fActivationAlpha.size() < i+1)
            fActivationAlpha.push_back(0.01);
         if (fActivationBeta.size() < i+1)
            fActivationBeta.push_back(0.0);
      } else if (fActivations[i] == "ThresholdedRelu") {
         if (fActivationAlpha.size() < i+1)
            fActivationAlpha.push_back(0.1);
         if (fActivationBeta.size() < i+1)
            fActivationBeta.push_back(0.0);
      } else if (fActivations[i] == "HardSigmoid") {
          if (fActivationAlpha.size() < i+1)
            fActivationAlpha.push_back(.2);
          if (fActivationBeta.size() < i+1)
            fActivationBeta.push_back(.5);
      } else if (fActivations[i] == "Elu") {
         if (fActivationAlpha.size() < i+1)
            fActivationAlpha.push_back(1.);
         if (fActivationBeta.size() < i+1)
            fActivationBeta.push_back(0.0);
      }
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
               sum[h] = B.GetData()[direction * 4 * fHiddenSize + h] +
                  B.GetData()[direction * 4 * fHiddenSize + fHiddenSize + h];
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
         for (size_t direction=0; direction < num_directions; direction++) {
            for (size_t batch=0; batch < batch_size; batch++) {
               for (size_t h=0; h < fHiddenSize; h++) {
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
         for (size_t direction=0; direction < num_directions; direction++) {
            for (size_t batch=0; batch < batch_size; batch++) {
               for (size_t h=0; h < fHiddenSize; h++) {
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
   if (fInputForget == 1) {
      forget_gate = new T[hidden_state_size];
   }
   T output_gate[hidden_state_size];
   T cell_input[hidden_state_size];
   // Set the cell state
   T cell_state[hidden_state_size];
   // Set the hidden state
   T * hidden_state = nullptr;
   if (fLayout == 0 && Y.GetShape().size() > 0) {
      hidden_state = Y.GetData();
   } else {
      hidden_state = new T[hidden_state_size];
   }

   T* gates[4] = {input_gate, forget_gate, output_gate, cell_input};

   for (size_t direction = 0; direction < num_directions; direction++) {
      for (size_t gate = 0; gate < 4; gate++) {
         if (gate == 1 && fInputForget == 0)
            continue;
         // feedforward = input * weight^T + bias
         char transA = 'N';
         char transB = 'T';
         int m = seq_length * batch_size;
         int n = fHiddenSize;
         int k = input_size;
         float alpha = 1.;
         float beta = 0.;
         size_t w_offset = (direction * 4 + gate) * fHiddenSize * input_size;
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

      for (size_t seq = 0; seq < seq_length; seq++) {
         // gate = 1.0 * gate + previous_hidden_state
         // Clip the elements of the gate into the range [-fClip, fClip]
         // gate = sigma(gate)
         // cell_state = input_gate o cell_input
         // cell_state = 1.0 * cell_state + forget_gate o previous_cell_state
         // Clip the elements of cell_state into the range [-fClip, fClip]
         // cell_state = phi(cell_state)
         // hidden_state = output_gate o cell_state
      }
   }

   // TODO LSTM with different sequence lengths

   // TODO copy hidden_state into Y, Y_h, and Y_c

   for (size_t gate = 0; gate < 4; gate++) {
      if (bias[gate])
         delete[] bias[gate];
   }

   if (fLayout == 1) {
      delete[] input;
      if (Y.GetShape().size() > 0)
         delete[] hidden_state;
   }
}


}
}
}

#endif
