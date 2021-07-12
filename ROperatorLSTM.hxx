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

   size_t seqLength = (fLayout == 0) ? X.GetShape()[0] : X.GetShape()[1];
   size_t batchSize = (fLayout == 0) ? X.GetShape()[1] : X.GetShape()[0];
   size_t inputSize = X.GetShape()[2];
   size_t numDirections = W.GetShape()[0];

   // Set the input
   T* Input = nullptr;
   if (fLayout == 0) {
      Input = X.GetData();
   } else {
      Input = new T[seqLength * batchSize * inputSize];
      for(size_t seq = 0; seq < seqLength; seq++) {
         for (size_t batch = 0; batch < batchSize; batch++) {
            for(size_t i = 0; i < inputSize; i++) {
               Input[seq * batchSize * inputSize + batch * inputSize + i] = X.GetData()[
                  batch * seqLength * inputSize + seq * inputSize + i];
            }
         }
      }
   }

   // Broadcasting the bias
   T* Bias = nullptr;
   if (B.GetShape().size() > 0) {
      Bias = new T[4 * numDirections * seqLength * batchSize * fHiddenSize];
      for (size_t gate = 0; gate < 4; gate++) {
         T sum[fHiddenSize];
         for (size_t direction = 0; direction < numDirections; direction++) {
            // Compute the sum of the gate-hidden bias and the hidden-hidden bias
            size_t offset = direction * 8 * fHiddenSize + gate * fHiddenSize;
            for (size_t h = 0; h < fHiddenSize; h++) {
               sum[h] = B.GetData()[offset + h] + B.GetData()[offset + h + 4 * fHiddenSize];
            }
            // Copy sum into bias
            for (size_t seq = 0; seq < seqLength; seq++) {
               for (size_t batch = 0; batch < batchSize; batch++) {
                  size_t biasOffset = gate * numDirections * seqLength * batchSize * fHiddenSize
                     + direction * seqLength * batchSize * fHiddenSize
                     + seq * batchSize * fHiddenSize + batch * fHiddenSize;
                  std::copy(sum, sum + fHiddenSize, Bias + biasOffset);
               }
            }
         }
      }
   }
   // Broadcasting the weight for peepholes
   T* Peephole = nullptr;
   if (P.GetShape().size() > 0) {
      if (batchSize == 1) {
         Peephole = P.GetData();
      } else {
         Peephole = new T[numDirections * 3 * batchSize * fHiddenSize];
         for (size_t direction = 0; direction < numDirections; direction++) {
            for (size_t gate = 0; gate < 3; gate++) {
               size_t pOffset = direction * 3 * fHiddenSize + gate * fHiddenSize;
               for (size_t batch = 0; batch < batchSize; batch++) {
                  size_t offset = direction * 3 * batchSize * fHiddenSize +
                     gate * batchSize * fHiddenSize + batch * fHiddenSize;
                  std::copy(P.GetData() + pOffset, P.GetData() + pOffset + fHiddenSize,
                     Peephole + offset);
               }
            }
         }
      }
   }
   // Set the initial hidden state
   T* InitialHiddenState = nullptr;
   if (initial_h.GetShape().size() > 0) {
      if (fLayout == 0) {
         InitialHiddenState = initial_h.GetData();
      } else {
         InitialHiddenState = new T[numDirections * batchSize * fHiddenSize];
         for (size_t direction = 0; direction < numDirections; direction++) {
            for (size_t batch = 0; batch < batchSize; batch++) {
               for (size_t h = 0; h < fHiddenSize; h++) {
                  InitialHiddenState[direction * batchSize * fHiddenSize + batch * fHiddenSize + h] =
                     initial_h.GetData()[batch * numDirections * fHiddenSize + direction * fHiddenSize
                        + h];
               }
            }
         }
      }
   }
   // Set the initial cell state
   T* InitialCellState = nullptr;
   if (initial_c.GetShape().size() > 0) {
      if (fLayout == 0) {
         InitialCellState = initial_c.GetData();
      } else {
         InitialCellState = new T[numDirections * batchSize * fHiddenSize];
         for (size_t direction = 0; direction < numDirections; direction++) {
            for (size_t batch = 0; batch < batchSize; batch++) {
               for (size_t h = 0; h < fHiddenSize; h++) {
                  InitialCellState[direction * batchSize * fHiddenSize + batch * fHiddenSize + h] =
                     initial_c.GetData()[batch * numDirections * fHiddenSize + direction * fHiddenSize
                        + h];
               }
            }
         }
      }
   }

   // Set the feedforward
   T FInputGate[seqLength * batchSize * fHiddenSize];
   T* FForgetGate = nullptr;
   if (fInputForget == 0) {
      FForgetGate = new T[seqLength * batchSize * fHiddenSize];
   }
   T FOutputGate[seqLength * batchSize * fHiddenSize];
   T FCellGate[seqLength * batchSize * fHiddenSize];
   // Set the gates
   size_t hiddenStateSize = seqLength * numDirections * batchSize * fHiddenSize;
   T InputGate[hiddenStateSize];
   T* ForgetGate = nullptr;
   if (fInputForget == 0) {
      ForgetGate = new T[hiddenStateSize];
   }
   T OutputGate[hiddenStateSize];
   T CellGate[hiddenStateSize];
   // Set the cell state
   T CellState[hiddenStateSize];
   // new cell state = h(cell_state)
   T NewCellState[hiddenStateSize];
   // Set the hidden state
   T * HiddenState = nullptr;
   if (fLayout == 0 && Y.GetShape().size() > 0) {
      HiddenState = Y.GetData();
   } else {
      HiddenState = new T[hiddenStateSize];
   }

   for (size_t direction = 0; direction < numDirections; direction++) {
      char transA = 'N';
      char transB = 'T';
      int m = seqLength * batchSize;
      int n = fHiddenSize;
      int k = inputSize;
      float alpha = 1.;
      float beta = 0.;
      // InputGate = Input * W_i^T
      size_t wiOffset = direction * 4 * fHiddenSize * inputSize;
      BLAS::sgemm_(&transB, &transA, &n, &m, &k, &alpha, W.GetData() + wiOffset, &k,
         Input, &k, &beta, FInputGate, &n);
      // OutputGate = Input * W_o^T
      size_t woOffset = direction * 4 * fHiddenSize * inputSize +
         2 * fHiddenSize * inputSize;
      BLAS::sgemm_(&transB, &transA, &n, &m, &k, &alpha, W.GetData() + woOffset, &k,
         Input, &k, &beta, FOutputGate, &n);
      // CellGate = Input * W_c^T
      size_t wcOffset = direction * 4 * fHiddenSize * inputSize +
         3 * fHiddenSize * inputSize;
      BLAS::sgemm_(&transB, &transA, &n, &m, &k, &alpha, W.GetData() + wcOffset, &k,
         Input, &k, &beta, FCellGate, &n);
      // ForgetGate = Input * W_f^T
      if (fInputForget == 0) {
         size_t wfOffset = direction * 4 * fHiddenSize * inputSize +
            fHiddenSize * inputSize;
         BLAS::sgemm_(&transB, &transA, &n, &m, &k, &alpha, W.GetData() + wfOffset, &k,
            Input, &k, &beta, FForgetGate, &n);
      }
      if (Bias) {
         int biasSize = seqLength * batchSize * fHiddenSize;
         int incx = 1;
         int incy = 1;
         // FInputGate += Bias_i
         size_t biOffset = direction * seqLength * batchSize * fHiddenSize;
         BLAS::saxpy_(&biasSize, &alpha, Bias + biOffset, &incx, FInputGate, &incy);
         // FOutputGate += Bias_o
         size_t boOffset = 2 * numDirections * seqLength * batchSize * fHiddenSize
            + direction * seqLength * batchSize * fHiddenSize;
         BLAS::saxpy_(&biasSize, &alpha, Bias + boOffset, &incx, FOutputGate, &incy);
         // FCellGate += Bias_c
         size_t bcOffset = 3 * numDirections * seqLength * batchSize * fHiddenSize
            + direction * seqLength * batchSize * fHiddenSize;
         BLAS::saxpy_(&biasSize, &alpha, Bias + bcOffset, &incx, FCellGate, &incy);
         // FForgetGate += Bias_f
         if (fInputForget == 0) {
            size_t bfOffset = numDirections * seqLength * batchSize * fHiddenSize
               + direction * seqLength * batchSize * fHiddenSize;
            BLAS::saxpy_(&biasSize, &alpha, Bias + bfOffset, &incx, FForgetGate, &incy);
         }
      }
      // Copy FInputGate, FOutputGate and FCellGate and FForgetGate into InputGate, OutputGate,
      //    CellGate and ForgetGate
      for (size_t seq = 0; seq < seqLength; seq++) {
         size_t ffOffset = seq * batchSize * fHiddenSize;
         size_t ffSize = batchSize * fHiddenSize;
         size_t gateOffset = seq * numDirections * batchSize * fHiddenSize +
            direction * batchSize * fHiddenSize;
         std::copy(FInputGate + ffOffset, FInputGate + ffOffset + ffSize, InputGate + gateOffset);
         std::copy(FOutputGate + ffOffset, FOutputGate + ffOffset + ffSize, OutputGate + gateOffset);
         std::copy(FCellGate + ffOffset, FCellGate + ffOffset + ffSize, CellGate + gateOffset);
         if (fInputForget == 0) {
            std::copy(FForgetGate + ffOffset, FForgetGate + ffOffset + ffSize, ForgetGate + gateOffset);
         }
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
               size_t initialhOffset = direction * batchSize * fHiddenSize;
               // InputGate += InitialHiddenState * R_i^T
               size_t riOffset = direction * 4 * fHiddenSize * fHiddenSize;
               BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + riOffset, &n,
                  InitialHiddenState + initialhOffset, &n, &alpha, InputGate + offset, &n);
               // OutputGate += InitialHiddenState * R_o^T
               size_t roOffset = direction * 4 * fHiddenSize * fHiddenSize +
                  2 * fHiddenSize * fHiddenSize;
               BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + roOffset, &n,
                  InitialHiddenState + initialhOffset, &n, &alpha, OutputGate + offset, &n);
               // CellGate += InitialHiddenState * R_c^T
               size_t rcOffset = direction * 4 * fHiddenSize * fHiddenSize +
                  3 * fHiddenSize * fHiddenSize;
               BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + rcOffset, &n,
                  InitialHiddenState + initialhOffset, &n, &alpha, CellGate + offset, &n);
               // ForgetGate += InitialHiddenState * R_f^T
               if (fInputForget == 0) {
                  size_t rfOffset = direction * 4 * fHiddenSize * fHiddenSize +
                     fHiddenSize * fHiddenSize;
                  BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + rfOffset, &n,
                     InitialHiddenState + initialhOffset, &n, &alpha, ForgetGate + offset, &n);
               }
            }
         } else {
            size_t previousOffset = (backward ? (index + 1) : (seq - 1)) * numDirections * batchSize *
               fHiddenSize + direction * batchSize * fHiddenSize;
            // InputGate += PreviousHiddenState * R_i^T
            size_t riOffset = direction * 4 * fHiddenSize * fHiddenSize;
            BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + riOffset, &n,
               HiddenState + previousOffset, &n, &alpha, InputGate + offset, &n);
            // OutputGate += PreviousHiddenState * R_o^T
            size_t roOffset = direction * 4 * fHiddenSize * fHiddenSize +
               2 * fHiddenSize * fHiddenSize;
            BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + roOffset, &n,
               HiddenState + previousOffset, &n, &alpha, OutputGate + offset, &n);
            // CellGate += PreviousHiddenState * R_c^T
            size_t rcOffset = direction * 4 * fHiddenSize * fHiddenSize +
               3 * fHiddenSize * fHiddenSize;
            BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + rcOffset, &n,
               HiddenState + previousOffset, &n, &alpha, CellGate + offset, &n);
            // ForgetGate += PreviousHiddenState * R_f^T
            if (fInputForget == 0) {
               size_t rfOffset = direction * 4 * fHiddenSize * fHiddenSize +
                  fHiddenSize * fHiddenSize;
               BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha, R.GetData() + rfOffset, &n,
                  HiddenState + previousOffset, &n, &alpha, ForgetGate + offset, &n);
            }
         }
         // Clip the elements of the cell gate into the range [-fClip, fClip]
         if (fClip > 0.) {
            for (size_t i = offset; i < offset + size; i++) {
               T x = (CellGate[i] > -fClip) ? CellGate[i] : -fClip;
               CellGate[i] = (x < fClip)? x : fClip;
            }
         }
         // Apply the activation function to the cell gate
         if (fActivations[direction * 3 + 1] == "Relu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (CellGate[i] < 0.)
                  CellGate[i] = 0.;
            }
         } else if (fActivations[direction * 3 + 1] == "Tanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float ex = exp(-2 * CellGate[i]);
               CellGate[i] = (1. - ex) / (1. + ex);
            }
         } else if (fActivations[direction * 3 + 1] == "Sigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               CellGate[i] = 1. / (1. + exp(-CellGate[i]));
            }
         } else if (fActivations[direction * 3 + 1] == "Affine") {
            for (size_t i = offset; i < offset + size; i++) {
               CellGate[i] = fActivationAlpha[direction * 3 + 1] * CellGate[i]
                  + fActivationBeta[direction * 3 + 1];
            }
         } else if (fActivations[direction * 3 + 1] == "LeakyRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (CellGate[i] < 0.) {
                  CellGate[i] = fActivationAlpha[direction * 3 + 1] * CellGate[i];
               }
            }
         } else if (fActivations[direction * 3 + 1] == "ThresholdRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (CellGate[i] < fActivationAlpha[direction * 3 + 1]) {
                  CellGate[i] = 0.;
               }
            }
         } else if (fActivations[direction * 3 + 1] == "ScaledTanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float x = exp(-2 * fActivationBeta[direction * 3 + 1] * CellGate[i]);
               CellGate[i] = fActivationAlpha[direction * 3 + 1] * (1. - x) / (1. + x);
            }
         } else if (fActivations[direction * 3 + 1] == "HardSigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               float a = fActivationAlpha[direction * 3 + 1] * CellGate[i] + fActivationBeta[direction * 3 + 1];
               float b = (a > 0.) ? a : 0.;
               CellGate[i] = (b < 1.) ? b : 1.;
            }
         } else if (fActivations[direction * 3 + 1] == "Elu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (CellGate[i] < 0.) {
                  CellGate[i] = fActivationAlpha[direction * 3 + 1] * (exp(CellGate[i] - 1.));
               }
            }
         } else if (fActivations[direction * 3 + 1] == "Softsign") {
            for (size_t i = offset; i < offset + size; i++) {
               CellGate[i] = CellGate[i] / (1. + abs(NewCellState[i]));
            }
         } else { // Softplus
            for (size_t i = offset; i < offset + size; i++) {
               CellGate[i] = log(1. + exp(CellGate[i]));
            }
         }
         // Peephole connections for the input gate and the forget gate
         if (Peephole) {
            if (seq == 0) {
               if (InitialCellState) {
                  size_t initialcOffset = direction * batchSize * fHiddenSize;
                  size_t piOffset = direction * 3 * batchSize * fHiddenSize;
                  // InputGate += InitialCellState o Peephole_i
                  for (size_t i = 0; i < size; i++) {
                     InputGate[i + offset] += InitialCellState[i + initialcOffset] * Peephole[i + piOffset];
                  }
                  // ForgetGate += InitialCellState o Peephole_f
                  if (fInputForget == 0) {
                     size_t pfOffset = direction * 3 * batchSize * fHiddenSize +
                        batchSize * fHiddenSize;
                     for (size_t i = 0; i < size; i++) {
                        ForgetGate[i + offset] += InitialCellState[i + initialcOffset] * Peephole[i + pfOffset];
                     }
                  }
               }
            } else {
               size_t cOffset = (backward ? (index + 1) : (seq - 1)) * numDirections * batchSize *
                  fHiddenSize + direction * batchSize * fHiddenSize;
               size_t piOffset = direction * 3 * batchSize * fHiddenSize;
               // InputGate += CellState o Peephole_i
               for (size_t i = 0; i < size; i++) {
                  InputGate[i + offset] += CellState[i + cOffset] * Peephole[i + piOffset];
               }
               // ForgetGate += CellState o Peephole_f
               if (fInputForget == 0) {
                  size_t pfOffset = direction * 3 * batchSize * fHiddenSize +
                     batchSize * fHiddenSize;
                  for (size_t i = 0; i < size; i++) {
                     ForgetGate[i + offset] += CellState[i + cOffset] * Peephole[i + pfOffset];
                  }
               }
            }
         }
         // Clip the elements of the input gate into the range [-fClip, fClip]
         if (fClip > 0.) {
            for (size_t i = offset; i < offset + size; i++) {
               T x = (InputGate[i] > -fClip) ? InputGate[i] : -fClip;
               InputGate[i] = (x < fClip)? x : fClip;
            }
         }
         // Apply the activation function to the input gate
         if (fActivations[direction * 3] == "Relu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (InputGate[i] < 0.)
                  InputGate[i] = 0.;
            }
         } else if (fActivations[direction * 3] == "Tanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float ex = exp(-2 * InputGate[i]);
               InputGate[i] = (1. - ex) / (1. + ex);
            }
         } else if (fActivations[direction * 3] == "Sigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               InputGate[i] = 1. / (1. + exp(-InputGate[i]));
            }
         } else if (fActivations[direction * 3] == "Affine") {
            for (size_t i = offset; i < offset + size; i++) {
               InputGate[i] = fActivationAlpha[direction * 3] * InputGate[i]
                  + fActivationBeta[direction * 3];
            }
         } else if (fActivations[direction * 3] == "LeakyRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (InputGate[i] < 0.) {
                  InputGate[i] = fActivationAlpha[direction * 3] * InputGate[i];
               }
            }
         } else if (fActivations[direction * 3] == "ThresholdRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (InputGate[i] < fActivationAlpha[direction * 3]) {
                  InputGate[i] = 0.;
               }
            }
         } else if (fActivations[direction * 3] == "ScaledTanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float x = exp(-2 * fActivationBeta[direction * 3] * InputGate[i]);
               InputGate[i] = fActivationAlpha[direction * 3] * (1. - x) / (1. + x);
            }
         } else if (fActivations[direction * 3] == "HardSigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               float a = fActivationAlpha[direction * 3] * InputGate[i] + fActivationBeta[direction * 3];
               float b = (a > 0.) ? a : 0.;
               InputGate[i] = (b < 1.) ? b : 1.;
            }
         } else if (fActivations[direction * 3] == "Elu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (InputGate[i] < 0.) {
                  InputGate[i] = fActivationAlpha[direction * 3] * (exp(InputGate[i] - 1.));
               }
            }
         } else if (fActivations[direction * 3] == "Softsign") {
            for (size_t i = offset; i < offset + size; i++) {
               InputGate[i] = InputGate[i] / (1. + abs(NewCellState[i]));
            }
         } else { // Softplus
            for (size_t i = offset; i < offset + size; i++) {
               InputGate[i] = log(1. + exp(InputGate[i]));
            }
         }

         if (fInputForget == 0) {
            // Clip the elements of the forget gate into the range [-fClip, fClip]
            if (fClip > 0.) {
               for (size_t i = offset; i < offset + size; i++) {
                  T x = (ForgetGate[i] > -fClip) ? ForgetGate[i] : -fClip;
                  ForgetGate[i] = (x < fClip)? x : fClip;
               }
            }
            // Apply the activation function to the forget gate
            if (fActivations[direction * 3] == "Relu") {
               for (size_t i = offset; i < offset + size; i++) {
                  if (ForgetGate[i] < 0.)
                     ForgetGate[i] = 0.;
               }
            } else if (fActivations[direction * 3] == "Tanh") {
               for (size_t i = offset; i < offset + size; i++) {
                  float ex = exp(-2 * ForgetGate[i]);
                  ForgetGate[i] = (1. - ex) / (1. + ex);
               }
            } else if (fActivations[direction * 3] == "Sigmoid") {
               for (size_t i = offset; i < offset + size; i++) {
                  ForgetGate[i] = 1. / (1. + exp(-ForgetGate[i]));
               }
            } else if (fActivations[direction * 3] == "Affine") {
               for (size_t i = offset; i < offset + size; i++) {
                  ForgetGate[i] = fActivationAlpha[direction * 3] * ForgetGate[i]
                     + fActivationBeta[direction * 3];
               }
            } else if (fActivations[direction * 3] == "LeakyRelu") {
               for (size_t i = offset; i < offset + size; i++) {
                  if (ForgetGate[i] < 0.) {
                     ForgetGate[i] = fActivationAlpha[direction * 3] * ForgetGate[i];
                  }
               }
            } else if (fActivations[direction * 3] == "ThresholdRelu") {
               for (size_t i = offset; i < offset + size; i++) {
                  if (ForgetGate[i] < fActivationAlpha[direction * 3]) {
                     ForgetGate[i] = 0.;
                  }
               }
            } else if (fActivations[direction * 3] == "ScaledTanh") {
               for (size_t i = offset; i < offset + size; i++) {
                  float x = exp(-2 * fActivationBeta[direction * 3] * ForgetGate[i]);
                  ForgetGate[i] = fActivationAlpha[direction * 3] * (1. - x) / (1. + x);
               }
            } else if (fActivations[direction * 3] == "HardSigmoid") {
               for (size_t i = offset; i < offset + size; i++) {
                  float a = fActivationAlpha[direction * 3] * ForgetGate[i] + fActivationBeta[direction * 3];
                  float b = (a > 0.) ? a : 0.;
                  ForgetGate[i] = (b < 1.) ? b : 1.;
               }
            } else if (fActivations[direction * 3] == "Elu") {
               for (size_t i = offset; i < offset + size; i++) {
                  if (ForgetGate[i] < 0.) {
                     ForgetGate[i] = fActivationAlpha[direction * 3] * (exp(ForgetGate[i] - 1.));
                  }
               }
            } else if (fActivations[direction * 3] == "Softsign") {
               for (size_t i = offset; i < offset + size; i++) {
                  ForgetGate[i] = ForgetGate[i] / (1. + abs(NewCellState[i]));
               }
            } else { // Softplus
               for (size_t i = offset; i < offset + size; i++) {
                  ForgetGate[i] = log(1. + exp(ForgetGate[i]));
               }
            }
         }
         // CellState = InputGate o CellGate
         for (size_t i = offset; i < offset + size; i++) {
            CellState[i] = InputGate[i] * CellGate[i];
         }
         if (fInputForget == 0) {
            if (seq == 0) {
               // CellState += ForgetGate o InitialCellState
               if (InitialCellState) {
                  for (size_t i = 0; i < size; i++) {
                     CellState[i + offset] += ForgetGate[i + offset] * InitialCellState[i];
                  }
               }
            } else {
               // CellState += ForgetGate o PreviousCellState
               size_t previousOffset = (backward ? (index + 1) : (seq - 1)) * numDirections * batchSize *
                  fHiddenSize + direction * batchSize * fHiddenSize;
               for (size_t i = 0; i < size; i++) {
                  CellState[i + offset] += ForgetGate[i + offset] * CellState[i + previousOffset];
               }
            }
         }
         if (Peephole) {
            // OutputGate += CellState o Peehole_o
            size_t pOffset = direction * 3 * batchSize * fHiddenSize +
               2 * batchSize * fHiddenSize;
            for (size_t i = 0; i < size; i++) {
               OutputGate[i + offset] += CellState[i + offset] * Peephole[i + pOffset];
            }
         }
         // Clip the elements of the output gate into the range [-fClip, fClip]
         if (fClip > 0.) {
            for (size_t i = offset; i < offset + size; i++) {
               T x = (OutputGate[i] > -fClip) ? OutputGate[i] : -fClip;
               OutputGate[i] = (x < fClip)? x : fClip;
            }
         }
         // Apply the activation function to the output gate
         if (fActivations[direction * 3] == "Relu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (OutputGate[i] < 0.)
                  OutputGate[i] = 0.;
            }
         } else if (fActivations[direction * 3] == "Tanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float ex = exp(-2 * OutputGate[i]);
               OutputGate[i] = (1. - ex) / (1. + ex);
            }
         } else if (fActivations[direction * 3] == "Sigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               OutputGate[i] = 1. / (1. + exp(-OutputGate[i]));
            }
         } else if (fActivations[direction * 3] == "Affine") {
            for (size_t i = offset; i < offset + size; i++) {
               OutputGate[i] = fActivationAlpha[direction * 3] * OutputGate[i]
                  + fActivationBeta[direction * 3];
            }
         } else if (fActivations[direction * 3] == "LeakyRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (OutputGate[i] < 0.) {
                  OutputGate[i] = fActivationAlpha[direction * 3] * OutputGate[i];
               }
            }
         } else if (fActivations[direction * 3] == "ThresholdRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (OutputGate[i] < fActivationAlpha[direction * 3]) {
                  OutputGate[i] = 0.;
               }
            }
         } else if (fActivations[direction * 3] == "ScaledTanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float x = exp(-2 * fActivationBeta[direction * 3] * OutputGate[i]);
               OutputGate[i] = fActivationAlpha[direction * 3] * (1. - x) / (1. + x);
            }
         } else if (fActivations[direction * 3] == "HardSigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               float a = fActivationAlpha[direction * 3] * OutputGate[i] + fActivationBeta[direction * 3];
               float b = (a > 0.) ? a : 0.;
               OutputGate[i] = (b < 1.) ? b : 1.;
            }
         } else if (fActivations[direction * 3] == "Elu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (OutputGate[i] < 0.) {
                  OutputGate[i] = fActivationAlpha[direction * 3] * (exp(OutputGate[i] - 1.));
               }
            }
         } else if (fActivations[direction * 3] == "Softsign") {
            for (size_t i = offset; i < offset + size; i++) {
               OutputGate[i] = OutputGate[i] / (1. + abs(NewCellState[i]));
            }
         } else { // Softplus
            for (size_t i = offset; i < offset + size; i++) {
               OutputGate[i] = log(1. + exp(OutputGate[i]));
            }
         }
         // Copy CellState into NewCellState
         std::copy(CellState + offset, CellState + offset + size, NewCellState + offset);
         // Clip the elements of NewCellState into the range [-fClip, fClip]
         if (fClip > 0.) {
            for (size_t i = offset; i < offset + size; i++) {
               T x = (NewCellState[i] > -fClip) ? NewCellState[i] : -fClip;
               NewCellState[i] = (x < fClip) ? x : fClip;
            }
         }
         // Apply the activation function to NewCellState
         if (fActivations[direction * 3 + 2] == "Relu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (NewCellState[i] < 0.)
                  NewCellState[i] = 0.;
            }
         } else if (fActivations[direction * 3 + 2] == "Tanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float ex = exp(-2 * NewCellState[i]);
               NewCellState[i] = (1. - ex) / (1. + ex);
            }
         } else if (fActivations[direction * 3 + 2] == "Sigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               NewCellState[i] = 1. / (1. + exp(-NewCellState[i]));
            }
         } else if (fActivations[direction * 3 + 2] == "Affine") {
            for (size_t i = offset; i < offset + size; i++) {
               NewCellState[i] = fActivationAlpha[direction * 3 + 2] * NewCellState[i]
                  + fActivationBeta[direction * 3 + 2];
            }
         } else if (fActivations[direction * 3 + 2] == "LeakyRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (NewCellState[i] < 0.) {
                  NewCellState[i] = fActivationAlpha[direction * 3 + 2] * NewCellState[i];
               }
            }
         } else if (fActivations[direction * 3 + 2] == "ThresholdRelu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (NewCellState[i] < fActivationAlpha[direction * 3 + 2]) {
                  NewCellState[i] = 0.;
               }
            }
         } else if (fActivations[direction * 3 + 2] == "ScaledTanh") {
            for (size_t i = offset; i < offset + size; i++) {
               float x = exp(-2 * fActivationBeta[direction * 3 + 2] * NewCellState[i]);
               NewCellState[i] = fActivationAlpha[direction * 3 + 2] * (1. - x) / (1. + x);
            }
         } else if (fActivations[direction * 3 + 2] == "HardSigmoid") {
            for (size_t i = offset; i < offset + size; i++) {
               float a = fActivationAlpha[direction * 3 + 2] * NewCellState[i] + fActivationBeta[direction * 3 + 2];
               float b = (a > 0.) ? a : 0.;
               NewCellState[i] = (b < 1.) ? b : 1.;
            }
         } else if (fActivations[direction * 3 + 2] == "Elu") {
            for (size_t i = offset; i < offset + size; i++) {
               if (NewCellState[i] < 0.) {
                  NewCellState[i] = fActivationAlpha[direction * 3 + 2] * (exp(NewCellState[i] - 1.));
               }
            }
         } else if (fActivations[direction * 3 + 2] == "Softsign") {
            for (size_t i = offset; i < offset + size; i++) {
               NewCellState[i] = NewCellState[i] / (1. + abs(NewCellState[i]));
            }
         } else { // Softplus
            for (size_t i = offset; i < offset + size; i++) {
               NewCellState[i] = log(1. + exp(NewCellState[i]));
            }
         }
         // HiddenState = OutputGate o NewCellState
         for (size_t i = offset; i < offset + size; i++) {
            HiddenState[i] = OutputGate[i] * NewCellState[i];
         }
      }
   }

   // LSTM with different sequence lengths
   if (sequence_lens.GetShape().size() > 0) {
      for (size_t seq = 0; seq < seqLength; seq++) {
         for (size_t batch = 0; batch < batchSize; batch++) {
            if (seq >= sequence_lens.GetData()[batch]) {
               for (size_t direction = 0; direction < numDirections; direction++) {
                  for (size_t h = 0; h < fHiddenSize; h++) {
                     size_t idx = seq * numDirections * batchSize * fHiddenSize
                        + direction * batchSize * fHiddenSize + batch * fHiddenSize + h;
                     CellState[idx] = 0.;
                     HiddenState[idx] = 0.;
                  }
               }
            }
         }
      }
   }

   // copy HiddenState into Y and Y_h, and copy cell_state into Y_c
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
      if (Y_c.GetShape().size() > 0) {
         if (sequence_lens.GetShape().size() > 0) {
            for (size_t direction = 0; direction < numDirections; direction++) {
               bool backward = (fDirection == "backward") || (direction == 1);
               for (size_t batch = 0; batch < batchSize; batch++) {
                  size_t seq = backward ? 0 : (sequence_lens.GetShape().size() > 0 ?
                     sequence_lens.GetData()[batch] - 1 : seqLength - 1);
                  size_t offset = seq * numDirections * batchSize * fHiddenSize
                     + direction * batchSize * fHiddenSize + batch * fHiddenSize;
                  size_t ycOffset = direction * batchSize * fHiddenSize
                     + batch * fHiddenSize;
                  std::copy(CellState + offset, HiddenState + offset + fHiddenSize,
                     Y_c.GetData() + ycOffset);
               }
            }
         } else {
            for (size_t direction = 0; direction < numDirections; direction++) {
               bool backward = (fDirection == "backward") || (direction == 1);
               size_t seq = backward ? 0 : seqLength - 1;
               size_t offset = seq * numDirections * batchSize * fHiddenSize +
                  direction * batchSize * fHiddenSize;
               size_t size = batchSize * fHiddenSize;
               size_t ycOffset = direction * batchSize * fHiddenSize;
               std::copy(CellState + offset, CellState + offset + size,
                  Y_c.GetData() + ycOffset);
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
      if (Y_c.GetShape().size() > 0) {
         for (size_t direction = 0; direction < numDirections; direction++) {
            bool backward = (fDirection == "backward") || (direction == 1);
            for (size_t batch = 0; batch < batchSize; batch++) {
               size_t seq = backward ? 0 : (sequence_lens.GetShape().size() > 0 ?
                  sequence_lens.GetData()[batch] - 1 : seqLength - 1);
               size_t offset = seq * numDirections * batchSize * fHiddenSize +
                  direction * batchSize * fHiddenSize + batch * fHiddenSize;
               size_t ycOffset = batch * numDirections * fHiddenSize +
                  direction * fHiddenSize;
               std::copy(CellState + offset, CellState + offset + fHiddenSize,
                  Y_c.GetData() + ycOffset);
            }
         }
      }
   }

   if (Bias)
      delete[] Bias;

   if (fInputForget == 0) {
      delete[] FForgetGate;
      delete[] ForgetGate;
   }

   if (fLayout == 1) {
      delete[] Input;
      delete[] InitialHiddenState;
      delete[] InitialCellState;
      if (Y.GetShape().size() == 0)
         delete[] HiddenState;
   }
   if (Peephole && batchSize > 1)
      delete[] Peephole;
}


}
}
}

#endif
