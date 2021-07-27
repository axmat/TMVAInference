#ifndef TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_GRU
#define TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_GRU

#include <TMVA/RTensor.hxx>
#include <algorithm>
#include <iostream>

#include "Blas.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

// Operator GRU
template<typename T> class ROperatorGRU {
private:
   /* Attributes */
   std::vector<T> fActivationAlpha;
   std::vector<T> fActivationBeta;
   std::vector<std::string> fActivations;
   T fClip;
   std::string fDirection;
   size_t fHiddenSize;
   size_t fLayout;
   size_t fLinearBeforeReset;

public:
   /* Constructor */
   ROperatorGRU(std::vector<T> activationAlpha, std::vector<T> activationBeta,
      std::vector<std::string> activations, T clip, std::string direction,
      size_t hiddenSize, size_t layout, size_t linearBeforeReset):
      fActivationAlpha(activationAlpha), fActivationBeta(activationBeta),
      fActivations(activations), fClip(clip), fDirection(direction),
      fHiddenSize(hiddenSize), fLayout(layout),
      fLinearBeforeReset(linearBeforeReset)
   {}

   /* Forward pass using Blas */
   void Forward_blas(RTensor<T> &X,
                     RTensor<T> &W,
                     RTensor<T> &R,
                     RTensor<T> &B,
                     RTensor<size_t> &sequence_lens,
                     RTensor<T> &initial_h,
                     RTensor<T> &Y,
                     RTensor<T> &Y_h);

};

template<typename T>
void ROperatorGRU<T>::Forward_blas(RTensor<T> &X,
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
   if (3 * fHiddenSize != W.GetShape()[1]) {
      throw std::runtime_error(
         "TMVA SOFIE - fHiddenSize must be equal to " + std::to_string(W.GetShape()[1] / 3));
   }
   if (fLayout > 1) {
      throw std::runtime_error(
          "TMVA SOFIE - Layout fLayout = " + std::to_string(fLayout) +
          " must be 0 (timewise) or 1 (batchwise)");
   }

   // Activation functions
   if (fActivations.empty()) {
      if (fDirection == "forward" || fDirection == "backward") {
         fActivations = {"Sigmoid", "Tanh"};
      } else {
         fActivations = {"Sigmoid", "Tanh", "Sigmoid", "Tanh"};
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
   T* Bias = nullptr;
   if (B.GetShape().size() > 0) {
      Bias = new T[numDirections * 6 * batchSize * fHiddenSize];
      for (size_t direction = 0; direction < numDirections; direction++) {
         for (size_t i = 0; i < 6; i++) {
            for (size_t batch = 0; batch < batchSize; batch++) {
               size_t bOffset = direction * 6 * fHiddenSize + i * fHiddenSize;
               size_t offset = direction * 6 * batchSize * fHiddenSize
                  + i * batchSize * fHiddenSize + batch * fHiddenSize;
               std::copy(B.GetData() + bOffset, B.GetData() + bOffset + fHiddenSize,
                  Bias + offset);
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

   // Set the feedforward
   size_t feedForwardSize = seqLength * batchSize * fHiddenSize;
   T FUpdateGate[feedForwardSize];
   T FResetGate[feedForwardSize];
   T FHiddenGate[feedForwardSize];
   // Set the gates
   size_t hiddenStateSize = seqLength * numDirections * batchSize * fHiddenSize;
   T UpdateGate[hiddenStateSize];
   T ResetGate[hiddenStateSize];
   T HiddenGate[hiddenStateSize];
   // set the hidden state
   T *HiddenState = nullptr;
   if (fLayout == 0 && Y.GetShape().size() > 0) {
      HiddenState = Y.GetData();
   } else {
      HiddenState = new T[hiddenStateSize];
   }

   T Feedback[batchSize * fHiddenSize];

   for (size_t direction = 0; direction < numDirections; direction++) {
      char transA = 'N';
      char transB = 'T';
      int m = seqLength * batchSize;
      int n = fHiddenSize;
      int k = inputSize;
      float alpha = 1.;
      float beta = 0.;
      // FUpdateGate = Input * W_z^T
      size_t wzOffset = direction * 3 * fHiddenSize * inputSize;
      BLAS::sgemm_(&transB, &transA, &n, &m, &k, &alpha, W.GetData() + wzOffset, &k,
         Input, &k, &beta, FUpdateGate, &n);
      // FResetGate = Input * W_r^T
      size_t wrOffset = direction * 3 * fHiddenSize * inputSize
         + fHiddenSize * inputSize;
      BLAS::sgemm_(&transB, &transA, &n, &m, &k, &alpha, W.GetData() + wrOffset, &k,
         Input, &k, &beta, FResetGate, &n);
      // FHiddenGate = Input * W_h^T
      size_t whOffset = direction * 3 * fHiddenSize * inputSize
         + 2 * fHiddenSize * inputSize;
      BLAS::sgemm_(&transB, &transA, &n, &m, &k, &alpha, W.GetData() + whOffset, &k,
         Input, &k, &beta, FHiddenGate, &n);
      if (Bias) {
         int bias_size = seqLength * batchSize * fHiddenSize;
         int incx = 1;
         int incy = 1;
         // Add the bias of the weight to FUpdateGate
         size_t wbzOffset = direction * 6 * batchSize * fHiddenSize;
         BLAS::saxpy_(&bias_size, &alpha, Bias + wbzOffset, &incx, FUpdateGate, &incy);
         // Add the bias of the recurrence to FUpdateGate
         size_t rbzOffset = direction * 3 * fHiddenSize * fHiddenSize + 3 * batchSize * fHiddenSize;
         BLAS::saxpy_(&bias_size, &alpha, Bias + rbzOffset, &incx, FUpdateGate, &incy);
         // Add the bias of the weight to FResetGate
         size_t wbrOffset = direction * 6 * batchSize * fHiddenSize
            + batchSize * fHiddenSize;
         BLAS::saxpy_(&bias_size, &alpha, Bias + wbrOffset, &incx, FResetGate, &incy);
         // Add the bias of the recurrence to FResetGate
         size_t rbrOffset = direction * 3 * fHiddenSize * fHiddenSize + fHiddenSize * fHiddenSize
            + 3 * batchSize * fHiddenSize;
         BLAS::saxpy_(&bias_size, &alpha, Bias + rbrOffset, &incx, FResetGate, &incy);
         // Add the bias of the weight to FHiddenGate
         size_t wbhOffset = direction * 6 * batchSize * fHiddenSize
            + 2 * batchSize * fHiddenSize;
         BLAS::saxpy_(&bias_size, &alpha, Bias + wbhOffset, &incx, FHiddenGate, &incy);
         if (fLinearBeforeReset == 0) {
            // Add the bias of te recurrence to FHiddenGate
            size_t rbhOffset = direction * 6 * batchSize * fHiddenSize
               + 5 * batchSize * fHiddenSize;
            BLAS::saxpy_(&bias_size, &alpha, Bias + rbhOffset, &incx, FHiddenGate, &incy);
         }
      }
      // Copy FUpdateGate, FResetGate and FHiddenGate to UpdateGate, ResetGate and HiddenGate
      for (size_t seq = 0; seq < seqLength; seq++) {
         size_t offset = seq * batchSize * fHiddenSize;
         size_t size = batchSize * fHiddenSize;
         size_t gateOffset = seq * numDirections * batchSize * fHiddenSize +
            direction * batchSize * fHiddenSize;
         std::copy(FUpdateGate + offset, FUpdateGate + offset + size, UpdateGate + gateOffset);
         std::copy(FResetGate + offset, FResetGate + offset + size, ResetGate + gateOffset);
         std::copy(FHiddenGate + offset, FHiddenGate + offset + size, HiddenGate + gateOffset);
      }

      bool backward = (fDirection == "backward") || (direction == 1);
      for (size_t seq = 0; seq < seqLength; seq++) {
         size_t index = backward ? seqLength - 1 - seq : seq;
         int m2 = batchSize;
         size_t offset = index * numDirections * batchSize * fHiddenSize
            + direction * batchSize * fHiddenSize;
         size_t size = batchSize * fHiddenSize;
         if (seq == 0) {
            if (InitialHiddenState) {
               size_t initialHOffset = direction * batchSize * fHiddenSize;
               // UpdateGate += InitialHiddenState * R_z^T
               size_t rUOffset = direction * 3 * fHiddenSize * fHiddenSize;
               BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + rUOffset, &n,
                  InitialHiddenState + initialHOffset, &n, &alpha, UpdateGate + offset, &n);
               // ResetGate += InitialHiddenState * R_r^T
               size_t rROffset = direction * 3 * fHiddenSize * fHiddenSize
                  + fHiddenSize * fHiddenSize;
               BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + rROffset, &n,
                  InitialHiddenState + initialHOffset, &n, &alpha, ResetGate + offset, &n);
            }
         } else {
            size_t previousOffset = (backward ? (index + 1) : (seq - 1)) * numDirections * 
               batchSize * fHiddenSize + direction * batchSize * fHiddenSize;
            // UpdateGate += PreviousHiddenState * R_z^T
            size_t rUOffset = direction * 3 * fHiddenSize * fHiddenSize;
            BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + rUOffset, &n,
               HiddenState + previousOffset, &n, &alpha, UpdateGate + offset, &n);
            // ResetGate += PreviousHiddenState * R_r^T
            size_t rROffset = direction * 3 * fHiddenSize * fHiddenSize
               + fHiddenSize * fHiddenSize;
            BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + rROffset, &n,
               HiddenState + previousOffset, &n, &alpha, ResetGate + offset, &n);
         }
         // Clip the elements of the update gate and the reset gate into the range [-fClip, fClip]
         if (fClip > 0.) {
            for (size_t i = offset; i < offset + size; i++) {
               T z = (UpdateGate[i] > -fClip) ? UpdateGate[i] : -fClip;
               UpdateGate[i] = (z < fClip)? z : fClip;
               T r = (ResetGate[i] > -fClip) ? ResetGate[i] : -fClip;
               ResetGate[i] = (r < fClip)? r : fClip;
            }
         }
         // Apply the activation function to the update gate and the reset gate
         if (fActivations[direction * 2] == "Relu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (UpdateGate[i] < 0.)
                  UpdateGate[i] = 0.;
               if (ResetGate[i] < 0.)
                  ResetGate[i] = 0.;
            }
         } else if (fActivations[direction * 2] == "Tanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float z = exp(-2 * UpdateGate[i]);
               UpdateGate[i] = (1. - z) / (1. + z);
               float r = exp(-2 * ResetGate[i]);
               ResetGate[i] = (1. - r) / (1. + r);
            }
         } else if (fActivations[direction * 2] == "Sigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               UpdateGate[i] = 1. / (1. + exp(-UpdateGate[i]));
               ResetGate[i] = 1. / (1. + exp(-ResetGate[i]));
            }
         } else {
            throw std::runtime_error("TMVA SOFIE - Activation function " + fActivations[direction * 2] +
               " not implemented.");
         }

         int feedbackSize = batchSize * fHiddenSize;
         int incx = 1;
         int incy = 1;
         if (fLinearBeforeReset == 0) {
            if (seq == 0) {
               // Feedback = ResetGate o InitialHiddenState
               if (InitialHiddenState) {
                  for (size_t i = 0; i < size; i++) {
                     Feedback[i] = ResetGate[i + offset] * InitialHiddenState[i];
                  }
               }
            } else {
               // Feedback = ResetGate o PreviousHiddenState
               size_t previousOffset = (backward ? (index + 1) : (seq - 1)) * numDirections *
                  batchSize * fHiddenSize + direction * batchSize * fHiddenSize;
               for (size_t i = 0; i < size; i++) {
                  Feedback[i] = ResetGate[i + offset] * HiddenState[i + previousOffset];
               }
            }
            // Feedback = Feedback * R_h^T
            size_t rh_offset = direction * 3 * fHiddenSize * fHiddenSize
               + 2 * fHiddenSize * fHiddenSize;
            BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + rh_offset, &n,
               Feedback, &n, &beta, Feedback, &n);
         } else { // fLinearBeforReset=1
            // Feedback = PreviousHiddenState * R_h^T
            size_t previousOffset = (backward ? (index + 1) : (seq - 1)) * numDirections *
               batchSize * fHiddenSize + direction * batchSize * fHiddenSize;
            size_t rh_offset = direction * 3 * fHiddenSize * fHiddenSize
               + 2 * fHiddenSize * fHiddenSize;
            BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + rh_offset, &n,
               HiddenState + previousOffset, &n, &beta, Feedback, &n);
            // Add the bias of te recurrence to Feedback
            size_t rbhOffset = direction * 6 * batchSize * fHiddenSize
               + 5 * batchSize * fHiddenSize;
            BLAS::saxpy_(&feedbackSize, &alpha, Bias + rbhOffset, &incx, Feedback, &incy);
            // Feedback = ResetGate o Feedback
            for (size_t i = 0; i < size; i++) {
               Feedback[i] *= ResetGate[i + offset];
            }
         }
         // HiddenGate = HiddenGate + Feedback
         BLAS::saxpy_(&feedbackSize, &alpha, Feedback, &incx, HiddenGate + offset, &incy);
         // Clip the elements of the hiddenGate into the range [-fClip, fClip]
         if (fClip > 0.) {
            for (size_t i = offset; i < offset + size; i++) {
               T x = (HiddenGate[i] > -fClip) ? HiddenGate[i] : -fClip;
               HiddenGate[i] = (x < fClip)? x : fClip;
            }
         }
         // Apply the activation function to the hidden gate
         if (fActivations[direction * 2 + 1] == "Relu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (HiddenGate[i] < 0.)
                  HiddenGate[i] = 0.;
            }
         } else if (fActivations[direction * 2 + 1] == "Tanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float ex = exp(-2 * HiddenGate[i]);
               HiddenGate[i] = (1. - ex) / (1. + ex);
            }
         } else if (fActivations[direction * 2 + 1] == "Sigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               HiddenGate[i] = 1. / (1. + exp(-HiddenGate[i]));
            }
         } else {
            throw std::runtime_error("TMVA - Activation function " + fActivations[direction * 2 + 1] +
               " not implemented.");
         }
         // HiddenState = (1 - UpdateGate) o HideenGate
         for (size_t i = offset; i < offset + size; i++) {
            HiddenState[i] = (1 - UpdateGate[i]) * HiddenGate[i];
         }
         if (seq == 0) {
            // HiddenState += UpdateGate o InitialHiddenState
            if (InitialHiddenState) {
               for (size_t i = 0; i < size; i++) {
                  HiddenState[i + offset] += UpdateGate[i + offset] + InitialHiddenState[i];
               }
            }
         } else {
            // HiddenState += UpdateGate o PreviousHiddenState
            size_t previousOffset = (backward ? (index + 1) : (seq - 1)) * numDirections *
               batchSize * fHiddenSize + direction * batchSize * fHiddenSize;
            for (size_t i = 0; i < size; i++) {
               HiddenState[i + offset] += UpdateGate[i + offset] * HiddenState[i + previousOffset];
            }
         }
      }
   }

   // padding the hidden state of GRU with different sequence lengths
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

   // Copy the hiddenState to Y and Y_h
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

   if (Bias)
      delete[] Bias;

   if (fLayout == 1) {
      if (Y.GetShape().size() == 0)
         delete[] HiddenState;
      if (InitialHiddenState)
         delete[] InitialHiddenState;
   }
}


}
}
}

#endif
