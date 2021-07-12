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

   size_t seqLength = (fLayout == 0) ? X.GetShape()[0] : X.GetShape()[1];
   size_t batchSize = (fLayout == 0) ? X.GetShape()[1] : X.GetShape()[0];
   size_t inputSize = X.GetShape()[2];
   size_t numDirections = W.GetShape()[0];

   // set the input
   T *Input = nullptr;
   if (fLayout == 0) {
      Input = X.GetData();
   } else {
      // reshape X
      Input = new T[seqLength * batchSize * inputSize];
      for (size_t seq = 0; seq < seqLength; seq++) {
         for (size_t batch = 0; batch < batchSize; batch++) {
            for (size_t i = 0; i < inputSize; i++) {
               Input[seq * batchSize * inputSize + batch * inputSize + i] =
                   X.GetData()[batch * seqLength * inputSize + seq * inputSize + i];
            }
         }
      }
   }

   // Broadcasting the bias
   T *Bias = nullptr;
   if (B.GetShape().size() > 0) {
      Bias = new T[numDirections * seqLength * batchSize * fHiddenSize];
      T sum[fHiddenSize];
      for (size_t direction = 0; direction < numDirections; direction++) {
         // Compute sum = B_{input,hidden} + B_{hidden,hidden}
         for (size_t h = 0; h < fHiddenSize; h++) {
            sum[h] = B.GetData()[direction * 2 * fHiddenSize + h] +
                B.GetData()[direction * 2 * fHiddenSize + fHiddenSize + h];
         }
         // Copy sum into Bias
         for (size_t seq = 0; seq < seqLength; seq++) {
            for (size_t batch = 0; batch < batchSize; batch++) {
               size_t biasOffset = direction * seqLength * batchSize * fHiddenSize +
                                    seq * batchSize * fHiddenSize + batch * fHiddenSize;
               std::copy(sum, sum + fHiddenSize, Bias + biasOffset);
            }
         }
      }
   }

   // set the initial hidden state
   T *InitialHiddenState = nullptr;
   if (initial_h.GetShape().size() > 0) {
      if (fLayout == 0) {
         InitialHiddenState = initial_h.GetData();
      } else {
         InitialHiddenState = new T[numDirections * batchSize * fHiddenSize];
         for (size_t direction = 0; direction < numDirections; direction++) {
            for (size_t batch = 0; batch < batchSize; batch++) {
               for (size_t h = 0; h < fHiddenSize; h++) {
                  InitialHiddenState[direction * batchSize * fHiddenSize +
                                       batch * fHiddenSize + h] =
                      initial_h.GetData()[batch * numDirections * fHiddenSize +
                                          direction * fHiddenSize + h];
               }
            }
         }
      }
   }

   T FeedForward[seqLength * batchSize * fHiddenSize];
   // set the hidden state
   T *HiddenState = nullptr;
   if (fLayout == 0 && Y.GetShape().size() > 0) {
      HiddenState = Y.GetData();
   } else {
      HiddenState = new T[seqLength * numDirections * batchSize * fHiddenSize];
   }

   for (size_t direction = 0; direction < numDirections; direction++) {
      bool backward = (fDirection == "backward") || (direction == 1);
      // Feedforward = input * W^T
      char transA = 'N';
      char transB = 'T';
      int m1 = seqLength * batchSize;
      int n = fHiddenSize;
      int k = inputSize;
      float alpha = 1.;
      float beta = 0.;
      size_t wOffset = direction * fHiddenSize * inputSize;
      BLAS::sgemm_(&transB, &transA, &n, &m1, &k, &alpha, W.GetData() + wOffset, &k,
                   Input, &k, &beta, FeedForward, &n);
      // Feedforward += Bias
      if (Bias) {
         int biasSize = seqLength * batchSize * fHiddenSize;
         int incx = 1;
         int incy = 1;
         size_t biasOffset = direction * seqLength * batchSize * fHiddenSize;
         BLAS::saxpy_(&biasSize, &alpha, Bias + biasOffset, &incx, FeedForward, &incy);
      }
      // Copy Feedforward into HiddenState
      for (size_t seq = 0; seq < seqLength; seq++) {
         size_t feedForwardOffset = seq * batchSize * fHiddenSize;
         size_t feedForwardSize = batchSize * fHiddenSize;
         size_t hOffset = seq * numDirections * batchSize * fHiddenSize +
                         direction * batchSize * fHiddenSize;
         std::copy(FeedForward + feedForwardOffset, FeedForward + feedForwardOffset + feedForwardSize,
                   HiddenState + hOffset);
      }

      for (size_t seq = 0; seq < seqLength; seq++) {
         size_t index = backward ? seqLength - 1 - seq : seq;
         // Compute HiddenState_{seq} = 1.0 * HiddenState_{seq} + HiddenState_{seq-1} * R^T
         // HiddenState_{seq} is the slice HiddenState[offset...offset + size]
         int m = batchSize;
         size_t offset = index * numDirections * batchSize * fHiddenSize +
                         direction * batchSize * fHiddenSize;
         size_t size = batchSize * fHiddenSize;
         if (seq == 0) {
            if (InitialHiddenState) {
               size_t rOffset = direction * fHiddenSize * fHiddenSize;
               size_t initialhOffset = direction * batchSize * fHiddenSize;
               BLAS::sgemm_(&transB, &transA, &n, &m, &n, &alpha, R.GetData() + rOffset, &n,
                  InitialHiddenState + initialhOffset, &n, &alpha, HiddenState + offset, &n);
            }
         } else {
            size_t rOffset = direction * fHiddenSize * fHiddenSize;
            size_t previousOffset = (backward ? (index + 1) : (seq - 1)) * numDirections * batchSize *
                                         fHiddenSize + direction * batchSize * fHiddenSize;
            BLAS::sgemm_(&transB, &transA, &n, &m, &n, &alpha, R.GetData() + rOffset, &n,
                         HiddenState + previousOffset, &n, &alpha, HiddenState + offset, &n);
         }
         // Clip the elements of HiddenState_{seq} into the range [-fClip, fClip]
         if (fClip > 0.) {
            for (size_t i = offset; i < offset + size; i++) {
               T x = (HiddenState[i] > -fClip) ? HiddenState[i] : -fClip;
               HiddenState[i] = (x < fClip) ? x : fClip;
            }
         }
         // Apply the activation function
         if (fActivations[direction] == "Relu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (HiddenState[i] < 0.)
                  HiddenState[i] = 0.;
            }
         } else if (fActivations[direction] == "Tanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float ex = exp(-2 * HiddenState[i]);
               HiddenState[i] = (1. - ex) / (1. + ex);
            }
         } else if (fActivations[direction] == "Sigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               HiddenState[i] = 1. / (1. + exp(-HiddenState[i]));
            }
         } else if (fActivations[direction] == "Affine") {
            for (size_t i = offset; i < offset + size; i++) {
               HiddenState[i] = fActivationAlpha[direction] * HiddenState[i] + fActivationBeta[direction];
            }
         } else if (fActivations[direction] == "LeakyRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (HiddenState[i] < 0.) {
                  HiddenState[i] = fActivationAlpha[direction] * HiddenState[i];
                  ;
               }
            }
         } else if (fActivations[direction] == "ThresholdRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (HiddenState[i] < fActivationAlpha[direction]) {
                  HiddenState[i] = 0.;
               }
            }
         } else if (fActivations[direction] == "ScaledTanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float x = exp(-2 * fActivationBeta[direction] * HiddenState[i]);
               HiddenState[i] = fActivationAlpha[direction] * (1. - x) / (1. + x);
            }
         } else if (fActivations[direction] == "HardSigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               float a = fActivationAlpha[direction] * HiddenState[i] + fActivationBeta[direction];
               float b = (a > 0.) ? a : 0.;
               HiddenState[i] = (b < 1.) ? b : 1.;
            }
         } else if (fActivations[direction] == "Elu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (HiddenState[i] < 0.) {
                  HiddenState[i] = fActivationAlpha[direction] * (exp(HiddenState[i] - 1.));
               }
            }
         } else if (fActivations[direction] == "Softsign") {
            for (size_t i = offset; i < offset + size; i++) {
               HiddenState[i] = HiddenState[i] / (1. + abs(HiddenState[i]));
            }
         } else { // Softplus
            for (size_t i = offset; i < offset + size; i++) {
               HiddenState[i] = log(1. + exp(HiddenState[i]));
            }
         }
      }
   }

   // padding the hidden state of RNN with different sequence lengths
   if (sequence_lens.GetShape().size() > 0) {
      for (size_t seq = 0; seq < seqLength; seq++) {
         for (size_t batch = 0; batch < batchSize; batch++) {
            if (seq >= sequence_lens.GetData()[batch]) {
               for (size_t direction = 0; direction < numDirections; direction++) {
                  for (size_t h = 0; h < fHiddenSize; h++) {
                     HiddenState[seq * numDirections * batchSize * fHiddenSize
                        + direction * batchSize * fHiddenSize + batch * fHiddenSize
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
            for (size_t direction = 0; direction < numDirections; direction++) {
               bool backward = (fDirection == "backward") || (direction == 1);
               for (size_t batch = 0; batch < batchSize; batch++) {
                  size_t seq = backward ? 0 : (sequence_lens.GetShape().size() > 0 ?
                     sequence_lens.GetData()[batch] - 1 : seqLength - 1);
                  size_t offset = seq * numDirections * batchSize * fHiddenSize
                     + direction * batchSize * fHiddenSize + batch * fHiddenSize;
                  size_t yhOffset = direction * batchSize * fHiddenSize
                     + batch * fHiddenSize;
                  std::copy(HiddenState + offset, HiddenState + offset + fHiddenSize,
                     Y_h.GetData() + yhOffset);
               }
            }
         } else {
            for (size_t direction = 0; direction < numDirections; direction++) {
               bool backward = (fDirection == "backward") || (direction == 1);
               size_t seq = backward ? 0 : seqLength - 1;
               size_t offset = seq * numDirections * batchSize * fHiddenSize +
                  direction * batchSize * fHiddenSize;
               size_t size = batchSize * fHiddenSize;
               size_t yhOffset = direction * batchSize * fHiddenSize;
               std::copy(HiddenState + offset, HiddenState + offset + size,
                  Y_h.GetData() + yhOffset);
            }
         }
      }
   } else { // fLayout=1
      if (Y.GetShape().size() > 0) {
         for (size_t seq = 0; seq < seqLength; seq++) {
            for (size_t direction = 0; direction < numDirections; direction++) {
               for (size_t batch = 0; batch < batchSize; batch++) {
                  size_t offset = seq * numDirections * batchSize * fHiddenSize +
                     direction * batchSize * fHiddenSize + batch * fHiddenSize;
                  size_t yOffset = batch * seqLength * numDirections * fHiddenSize +
                     seq * numDirections * fHiddenSize + direction * fHiddenSize;
                  std::copy(HiddenState + offset, HiddenState + offset + fHiddenSize,
                            Y.GetData() + yOffset);
               }
            }
         }
      }
      if (Y_h.GetShape().size() > 0) {
         for (size_t direction = 0; direction < numDirections; direction++) {
            bool backward = (fDirection == "backward") || (direction == 1);
            for (size_t batch = 0; batch < batchSize; batch++) {
               size_t seq = backward ? 0 : (sequence_lens.GetShape().size() > 0 ?
                  sequence_lens.GetData()[batch] - 1 : seqLength - 1);
               size_t offset = seq * numDirections * batchSize * fHiddenSize +
                  direction * batchSize * fHiddenSize + batch * fHiddenSize;
               size_t yhOffset = batch * numDirections * fHiddenSize +
                  direction * fHiddenSize;
               std::copy(HiddenState + offset, HiddenState + offset + fHiddenSize,
                  Y_h.GetData() + yhOffset);
            }
         }
      }
   }

   if (Bias) {
      delete[] Bias;
   }
   if (fLayout == 1) {
      delete[] Input;
      delete[] InitialHiddenState;
   }
   if (fLayout == 1 || Y.GetShape().size() == 0)
      delete[] HiddenState;
}

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
