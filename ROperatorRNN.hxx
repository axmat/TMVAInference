#ifndef TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_RNN
#define TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_RNN

#include <TMVA/RTensor.hxx>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "Blas.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

// Operator RNN
template <typename T> class ROperatorRNN {
 private:
   /* Attributes */
   std::vector<T> fActivationAlpha;
   std::vector<T> fActivationBeta;
   std::vector<std::string> fActivations;
   T fClip;
   std::string fDirection;
   std::size_t fHiddenSize;
   std::size_t fLayout;

 public:
   /* Constructor */
   ROperatorRNN(std::vector<T> activationAlpha, std::vector<T> activationBeta,
                std::vector<std::string> activations, T clip,
                std::string direction, std::size_t hiddenSize,
                std::size_t layout)
       : fActivationAlpha(activationAlpha), fActivationBeta(activationBeta),
         fActivations(activations), fClip(clip), fDirection(direction),
         fHiddenSize(hiddenSize), fLayout(layout) {}

   /* Forward pass using Blas */
   void Forward_blas(RTensor<T> &X,
                     RTensor<T> &W,
                     RTensor<T> &R,
                     RTensor<T> &B,
                     RTensor<size_t> &sequence_lens,
                     RTensor<T> &initial_h,
                     RTensor<T> &Y,
                     RTensor<T> &Yh);
};

template <typename T>
void ROperatorRNN<T>::Forward_blas(RTensor<T> &X,
                                   RTensor<T> &W,
                                   RTensor<T> &R,
                                   RTensor<T> &B,
                                   RTensor<size_t> &sequence_lens,
                                   RTensor<T> &initial_h,
                                   RTensor<T> &Y,
                                   RTensor<T> &Y_h) {
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
   if (fDirection != "forward" && fDirection != "backward" &&
       fDirection != "bidirectional") {
      throw std::runtime_error(
          "TMVA SOFIE - Invalid RNN direction fDirection = " + fDirection);
   }
   if (fHiddenSize != W.GetShape()[1]) {
      throw std::runtime_error(
         "TMVA SOFIE - fHiddenSize must be equal to " + std::to_string(W.GetShape()[1]));
   }
   if (fLayout > 1) {
      throw std::runtime_error(
          "TMVA SOFIE - Layout fLayout = " + std::to_string(fLayout) +
          " must be 0 (timewise) or 1 (batchwise)");
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
         if (fActivationAlpha.size() < i + 1)
            fActivationAlpha.push_back(0.01);
         if (fActivationBeta.size() < i + 1)
            fActivationBeta.push_back(0.0);
      } else if (fActivations[i] == "ThresholdedRelu") {
         if (fActivationAlpha.size() < i + 1)
            fActivationAlpha.push_back(0.1);
         if (fActivationBeta.size() < i + 1)
            fActivationBeta.push_back(0.0);
      } else if (fActivations[i] == "HardSigmoid") {
         if (fActivationAlpha.size() < i + 1)
            fActivationAlpha.push_back(.2);
         if (fActivationBeta.size() < i + 1)
            fActivationBeta.push_back(.5);
      } else if (fActivations[i] == "Elu") {
         if (fActivationAlpha.size() < i + 1)
            fActivationAlpha.push_back(1.);
         if (fActivationBeta.size() < i + 1)
            fActivationBeta.push_back(0.);
      }
   }

   size_t seq_length = (fLayout == 0) ? X.GetShape()[0] : X.GetShape()[1];
   size_t batch_size = (fLayout == 0) ? X.GetShape()[1] : X.GetShape()[0];
   size_t input_size = X.GetShape()[2];
   size_t num_directions = W.GetShape()[0];

   // set the input
   T *input = nullptr;
   if (fLayout == 0) {
      input = X.GetData();
   } else {
      // reshape X
      input = new T[seq_length * batch_size * input_size];
      for (size_t seq = 0; seq < seq_length; seq++) {
         for (size_t batch = 0; batch < batch_size; batch++) {
            for (size_t i = 0; i < input_size; i++) {
               input[seq * batch_size * input_size + batch * input_size + i] =
                   X.GetData()[batch * seq_length * input_size + seq * input_size + i];
            }
         }
      }
   }

   // Broadcasting the bias
   T *bias = nullptr;
   if (B.GetShape().size() > 0) {
      bias = new T[num_directions * seq_length * batch_size * fHiddenSize];
      T sum[fHiddenSize];
      for (size_t direction = 0; direction < num_directions; direction++) {
         // Compute sum = bias_xh + bias_hh
         for (size_t h = 0; h < fHiddenSize; h++) {
            sum[h] = B.GetData()[direction * 2 * fHiddenSize + h] +
                B.GetData()[direction * 2 * fHiddenSize + fHiddenSize + h];
         }
         // Copy sum into bias
         for (size_t seq = 0; seq < seq_length; seq++) {
            for (size_t batch = 0; batch < batch_size; batch++) {
               size_t bias_offset = direction * seq_length * batch_size * fHiddenSize +
                                    seq * batch_size * fHiddenSize + batch * fHiddenSize;
               std::copy(sum, sum + fHiddenSize, bias + bias_offset);
            }
         }
      }
   }

   // set the initial hidden state
   T *initial_hidden_state = nullptr;
   if (initial_h.GetShape().size() > 0) {
      if (fLayout == 0) {
         initial_hidden_state = initial_h.GetData();
      } else {
         initial_hidden_state = new T[num_directions * batch_size * fHiddenSize];
         for (size_t direction = 0; direction < num_directions; direction++) {
            for (size_t batch = 0; batch < batch_size; batch++) {
               for (size_t h = 0; h < fHiddenSize; h++) {
                  initial_hidden_state[direction * batch_size * fHiddenSize +
                                       batch * fHiddenSize + h] =
                      initial_h.GetData()[batch * num_directions * fHiddenSize +
                                          direction * fHiddenSize + h];
               }
            }
         }
      }
   }

   T feedforward[seq_length * batch_size * fHiddenSize];
   // set the hidden state
   T *hidden_state = nullptr;
   if (fLayout == 0 && Y.GetShape().size() > 0) {
      hidden_state = Y.GetData();
   } else {
      hidden_state = new T[seq_length * num_directions * batch_size * fHiddenSize];
   }

   for (size_t direction = 0; direction < num_directions; direction++) {
      bool backward = (fDirection == "backward") || (direction == 1);
      // feedforward = input * W^T + bias
      char transA = 'N';
      char transB = 'T';
      int m1 = seq_length * batch_size;
      int n = fHiddenSize;
      int k = input_size;
      float alpha = 1.;
      float beta = 0.;
      size_t w_offset = direction * fHiddenSize * input_size;
      BLAS::sgemm_(&transB, &transA, &n, &m1, &k, &alpha, W.GetData() + w_offset, &k,
                   input, &k, &beta, feedforward, &n);
      if (bias) {
         int bias_size = seq_length * batch_size * fHiddenSize;
         int incx = 1;
         int incy = 1;
         size_t bias_offset = direction * seq_length * batch_size * fHiddenSize;
         BLAS::saxpy_(&bias_size, &alpha, bias + bias_offset, &incx, feedforward, &incy);
      }
      // Copy feedforward into hidden_state
      for (size_t seq = 0; seq < seq_length; seq++) {
         size_t feedforward_offset = seq * batch_size * fHiddenSize;
         size_t feedforward_size = batch_size * fHiddenSize;
         size_t offset = seq * num_directions * batch_size * fHiddenSize +
                         direction * batch_size * fHiddenSize;
         std::copy(feedforward + feedforward_offset, feedforward + feedforward_offset + feedforward_size,
                   hidden_state + offset);
      }

      for (size_t seq = 0; seq < seq_length; seq++) {
         size_t index = backward ? seq_length - 1 - seq : seq;
         // Compute hidden_state_{seq} = 1.0 * hidden_state_{seq} + hidden_state_{seq-1} * R^T
         // hidden_state_{seq} is the slice hidden_state[offset...offset + size]
         int m2 = batch_size;
         size_t offset = index * num_directions * batch_size * fHiddenSize +
                         direction * batch_size * fHiddenSize;
         size_t size = batch_size * fHiddenSize;
         if (seq == 0) {
            if (initial_hidden_state) {
               size_t r_offset = direction * fHiddenSize * fHiddenSize;
               size_t initial_hidden_state_offset = direction * batch_size * fHiddenSize;
               BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + r_offset, &n,
                            initial_hidden_state + initial_hidden_state_offset, &n, &alpha,
                            hidden_state + offset, &n);
            }
         } else {
            size_t r_offset = direction * fHiddenSize * fHiddenSize;
            size_t previous_offset = (backward ? (index + 1) : (seq - 1)) * num_directions * batch_size *
                                         fHiddenSize + direction * batch_size * fHiddenSize;
            BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + r_offset, &n,
                         hidden_state + previous_offset, &n, &alpha, hidden_state + offset, &n);
         }
         // Clip the elements of hidden_state_{seq} into the range [-fClip, fClip]
         if (fClip > 0.) {
            for (size_t i = offset; i < offset + size; i++) {
               T x = (hidden_state[i] > -fClip) ? hidden_state[i] : -fClip;
               hidden_state[i] = (x < fClip) ? x : fClip;
            }
         }
         // Apply the activation function
         if (fActivations[direction] == "Relu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (hidden_state[i] < 0.)
                  hidden_state[i] = 0.;
            }
         } else if (fActivations[direction] == "Tanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float ex = exp(-2 * hidden_state[i]);
               hidden_state[i] = (1. - ex) / (1. + ex);
            }
         } else if (fActivations[direction] == "Sigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               hidden_state[i] = 1. / (1. + exp(-hidden_state[i]));
            }
         } else if (fActivations[direction] == "Affine") {
            for (size_t i = offset; i < offset + size; i++) {
               hidden_state[i] = fActivationAlpha[direction] * hidden_state[i] + fActivationBeta[direction];
            }
         } else if (fActivations[direction] == "LeakyRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (hidden_state[i] < 0.) {
                  hidden_state[i] = fActivationAlpha[direction] * hidden_state[i];
                  ;
               }
            }
         } else if (fActivations[direction] == "ThresholdRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (hidden_state[i] < fActivationAlpha[direction]) {
                  hidden_state[i] = 0.;
               }
            }
         } else if (fActivations[direction] == "ScaledTanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float x = exp(-2 * fActivationBeta[direction] * hidden_state[i]);
               hidden_state[i] = fActivationAlpha[direction] * (1. - x) / (1. + x);
            }
         } else if (fActivations[direction] == "HardSigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               float a = fActivationAlpha[direction] * hidden_state[i] + fActivationBeta[direction];
               float b = (a > 0.) ? a : 0.;
               hidden_state[i] = (b < 1.) ? b : 1.;
            }
         } else if (fActivations[direction] == "Elu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (hidden_state[i] < 0.) {
                  hidden_state[i] = fActivationAlpha[direction] * (exp(hidden_state[i] - 1.));
               }
            }
         } else if (fActivations[direction] == "Softsign") {
            for (size_t i = offset; i < offset + size; i++) {
               hidden_state[i] = hidden_state[i] / (1. + abs(hidden_state[i]));
            }
         } else { // Softplus
            for (size_t i = offset; i < offset + size; i++) {
               hidden_state[i] = log(1. + exp(hidden_state[i]));
            }
         }
      }
   }

   // padding the hidden state of RNN with different sequence lengths
   if (sequence_lens.GetShape().size() > 0) {
      for (size_t seq = 0; seq < seq_length; seq++) {
         for (size_t batch = 0; batch < batch_size; batch++) {
            if (seq >= sequence_lens.GetData()[batch]) {
               for (size_t direction = 0; direction < num_directions; direction++) {
                  for (size_t h = 0; h < fHiddenSize; h++) {
                     hidden_state[seq * num_directions * batch_size * fHiddenSize
                        + direction * batch_size * fHiddenSize + batch * fHiddenSize
                        + h] = .0;
                  }
               }
            }
         }
      }
   }

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
   }

   if (bias) {
      delete[] bias;
   }
   if (fLayout == 1) {
      delete[] input;
      delete[] initial_hidden_state;
   }
   if (fLayout == 1 || Y.GetShape().size() == 0)
      delete[] hidden_state;
}

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
