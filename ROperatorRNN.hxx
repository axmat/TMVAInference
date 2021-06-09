#ifndef TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_RNN
#define TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_RNN

#include <TMVA/RTensor.hxx>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "Blas.hxx"
#include "./testROperator.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

// Operator RNN
template<typename T> class ROperatorRNN {
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
      std::vector<std::string> activations, T clip, std::string direction,
      std::size_t hiddenSize, std::size_t layout):
      fActivationAlpha(activationAlpha), fActivationBeta(activationBeta),
      fActivations(activations), fClip(clip), fDirection(direction),
      fHiddenSize(hiddenSize), fLayout(layout)
   {}

   /* Forward pass using Blas */
   void Forward_blas(RTensor<T> &X,
                     RTensor<T> &W,
                     RTensor<T> &R,
                     RTensor<T> &B,
                     RTensor<T> &SequenceLens,
                     RTensor<T> &initial_h,
                     RTensor<T> &Y,
                     RTensor<T> &Yh);

};

template<typename T>
void ROperatorRNN<T>::Forward_blas(RTensor<T> &X,
                                   RTensor<T> &W,
                                   RTensor<T> &R,
                                   RTensor<T> &B,
                                   RTensor<T> &sequence_lens,
                                   RTensor<T> &initial_h,
                                   RTensor<T> &Y,
                                   RTensor<T> &Y_h) {
   // Activation functions
   if (fActivations.empty()) {
      if (fDirection == "forward" || fDirection == "backward") {
         fActivations = {"Tanh"};
      } else if (fDirection == "bidirectional") {
         fActivations = {"Tanh", "Tanh"};
      } else {
         throw
            std::runtime_error("TMVA SOFIE - Invalid fDirection = " + fDirection);
      }
   }
   // Default values of alpha and beta
   for (size_t i=0; i < fActivations.size(); i++) {
      if (fActivations[i] == "LeakyRelu") {
         if (fActivationAlpha.size() < i+1)
            fActivationAlpha.push_back(0.01);
      } else if (fActivations[i] == "ThresholdedRelu") {
         if (fActivationAlpha.size() < i+1) {
            fActivationAlpha.push_back(0.1);
         }
      } else if (fActivations[i] == "HardSigmoid") {
          if (fActivationAlpha.size() < i+1 && fActivationBeta.size() < i+1) {
            fActivationAlpha.push_back(.2);
            fActivationBeta.push_back(.5);
         }
      } else if (fActivations[i] == "Elu") {
         if (fActivationAlpha.size() < i+1) {
            fActivationAlpha.push_back(1.);
         }
      }
   }
   /////////////////////////////////////////////////////////////////////////////////////////////////
   size_t seq_length;
   size_t batch_size;
   size_t input_size;
   if (fLayout == 0) {
      seq_length = X.GetShape()[0];
      batch_size = X.GetShape()[1];
      input_size = X.GetShape()[2];
   } else if (fLayout == 1) {
      batch_size = X.GetShape()[0];
      seq_length = X.GetShape()[1];
      input_size = X.GetShape()[2];
   } else {
      throw
         std::runtime_error("TMVA SOFIE - Invalid fLayout");
   }
   std::size_t num_directions = W.GetShape()[0];
   std::size_t hidden_size = W.GetShape()[1];
   // sequence_lens of shape batch_size
   if (sequence_lens.GetSize() == 0) {
      sequence_lens = RTensor<T>({batch_size});
      std::fill(sequence_lens.begin(), sequence_lens.end(), seq_length);
   }
   /////////////////////////////////////////////////////////////////////////////////////////////////
   // Broadcasting the bias B
   T* bias = nullptr;
   if (B.GetSize() > 0) {
      bias = new T[num_directions * seq_length * batch_size * hidden_size];
      T* sum_bias = new T[hidden_size];
      for (size_t direction = 0; direction < num_directions; direction++) {
         // Compute bias_xh + bias_hh
         for (size_t hid = 0; hid < hidden_size; hid++) {
            sum_bias[hid] = B.GetData()[direction * 2*hidden_size + hid] + B.GetData()[direction * 2* hidden_size + hidden_size + hid];
         }
         // Copy sum_bias into bias
         for (size_t seq = 0; seq < seq_length; seq++) {
            for (size_t batch = 0; batch < batch_size; batch++) {
               std::copy(sum_bias, sum_bias + hidden_size,
                  bias + direction * seq_length * batch_size * hidden_size + seq * batch_size * hidden_size + batch * hidden_size
);
            }
         }
      }
      delete[] sum_bias;
   }
   // set the input
   T* input = nullptr;
   if (fLayout == 0) {
      input = X.GetData();
   } else {
      // reshape x
      input = new T[seq_length * batch_size * input_size];
      for(size_t seq = 0; seq < seq_length; seq++) {
         for (size_t batch = 0; batch < batch_size; batch++) {
            for(size_t i = 0; i < input_size; i++) {
               input[seq * batch_size * input_size + batch * input_size + i] = X(batch * seq_length * input_size + seq * input_size + i);
            }
         }
      }
   }
   T* feedforward = new T[seq_length * batch_size * hidden_size];
   // set the hidden state
   T * hidden_state = nullptr;
   if (fLayout == 0 && Y.GetSize() > 0) {
      hidden_state = Y.GetData();
   } else {
      hidden_state = new T[seq_length * num_directions * batch_size * hidden_size];
   }
   // set the initial hidden state
   T* initial_hidden_state = nullptr;
   if (initial_h.GetSize() > 0) {
      if (fLayout == 0) {
         initial_hidden_state = initial_h.GetData();
      } else {
         initial_hidden_state = new T[num_directions * batch_size * hidden_size];
         for (size_t direction=0; direction < num_directions; direction++) {
            for (size_t batch=0; batch < batch_size; batch++) {
               for (size_t hid=0; hid < hidden_size; hid++) {
                  initial_hidden_state[direction * batch_size * hidden_size + batch * hidden_size + hid] =
                     initial_h.GetData()[batch * num_directions * hidden_size + direction * hidden_size
                        + hid];
               }
            }
         }
      }
   }

   for (size_t direction = 0; direction < num_directions; direction++) {
      bool backward = (fDirection == "backward") || (direction == 1);
      // input * W^T
      char transA = 'N';
      char transB = 'T';
      int m1 = seq_length * batch_size;
      int n = hidden_size;
      int k = input_size;
      float alpha = 1.;
      float beta = 0.;
      BLAS::sgemm_(&transB, &transA, &n, &m1, &k, &alpha,
         W.GetData() + direction * hidden_size * input_size, &k,
         input, &k, &beta, feedforward, &n);
      // Add bias
      int feedforward_size = seq_length * batch_size * hidden_size;
      int incx = 1;
      int incy = 1;
      BLAS::saxpy_(&feedforward_size, &alpha, bias + direction * seq_length * batch_size * hidden_size, &incx, feedforward, &incy);
      // Copy feedforward to hidden_state
      //size_t hseq_size = batch_size * hidden_size;
      for (size_t seq = 0; seq < seq_length; seq++) {
         std::copy(feedforward + seq * batch_size * hidden_size,
            feedforward + (seq + 1) * batch_size * hidden_size,
            hidden_state + seq * num_directions * batch_size * hidden_size + direction * batch_size * hidden_size);
      }

      for (size_t seq = 0; seq < seq_length; seq++) {
         size_t idx = backward? seq_length - 1 - seq : seq;
         // Compute h_seq = 1.0 * h_seq + h_seq-1 * R^T
         // h_seq is the slice hidden_state[hseq_start...hseq_end]
         int m2 = batch_size;
         size_t hseq_start = idx * num_directions * batch_size * hidden_size + direction * batch_size * hidden_size;
         size_t hseq_end = idx * num_directions * batch_size * hidden_size + (direction + 1) * batch_size * hidden_size;
         if (seq == 0) {
            if (initial_hidden_state) {
               BLAS::sgemm_(&transB, &transA, &n, &m2 , &n, &alpha,
                  R.GetData() + direction * hidden_size * hidden_size, &n,
                  initial_hidden_state + direction * batch_size * hidden_size, &n, &alpha,
                  hidden_state + hseq_start, &n);
            }
         } else {
            size_t hprevseq_start = (backward? (idx + 1): (seq - 1)) * num_directions * batch_size * hidden_size + direction * batch_size * hidden_size;
            BLAS::sgemm_(&transB, &transA, &n, &m2, &n, &alpha,
               R.GetData() + direction * hidden_size * hidden_size, &n,
               hidden_state + hprevseq_start, &n, &alpha,
               hidden_state + hseq_start, &n);
         }
         // Apply the activation function
         if (fActivations[direction] == "Relu") {
            for (size_t i = hseq_start; i < hseq_end; i++) {
               if (hidden_state[i] > 0)
                  hidden_state[i] = 0.;
            }
         } else if (fActivations[direction] == "Tanh") {
            for (size_t i = hseq_start; i < hseq_end; i++) {
               float ex = exp(-2 * hidden_state[i]);
               hidden_state[i] = (1. - ex) / (1. + ex);
            }
         } else if (fActivations[direction] == "Sigmoid") {
            for (size_t i = hseq_start; i < hseq_end; i++) {
               hidden_state[i] = 1. / (1. + exp(-hidden_state[i]));
            }
         } else if (fActivations[direction] == "Affine") {
            for (size_t i = hseq_start; i < hseq_end; i++) {
               hidden_state[i] = fActivationAlpha[0] * hidden_state[i] + fActivationBeta[0];
            }
         } else if (fActivations[direction] == "LeakyRelu") {
            for (size_t i = hseq_start; i < hseq_end; i++) {
               if (hidden_state[i] < 0.) {
                  hidden_state[i] = fActivationAlpha[0] * hidden_state[i];;
               }
            }
         } else if (fActivations[direction] == "ThresholdRelu") {
            for (size_t i = hseq_start; i < hseq_end; i++) {
               if (hidden_state[i] < 0. ) {
                  hidden_state[i] = 0.;
               }
            }
         } else if (fActivations[direction] == "ScaledTanh") {
            for (size_t i = hseq_start; i < hseq_end; i++) {
               float x = exp(-2 * fActivationBeta[0] * hidden_state[i]);
               hidden_state[i] = fActivationAlpha[0] * (1. - x) / (1. + x);
            }
         } else if (fActivations[direction] == "HardSigmoid") {
            for (size_t i = hseq_start; i < hseq_end; i++) {
               float x = ((fActivationAlpha[0] * hidden_state[i] + fActivationBeta[0]) > 0)?
                  fActivationAlpha[0] * hidden_state[i] + fActivationBeta[0] : 0.;
               hidden_state[i] = (x < 1.)? x : 1.;
            }
         } else if (fActivations[direction] == "Elu") {
            for (size_t i = hseq_start; i < hseq_end; i++) {
               if (hidden_state[i] < 0) {
                  hidden_state[i] = fActivationAlpha[0] * (exp(hidden_state[i] - 1.));
               }
            }
         } else if (fActivations[direction] == "Softsign") {
            for (size_t i = hseq_start; i < hseq_end; i++) {
               hidden_state[i] = hidden_state[i] / (1. + abs(hidden_state[i]));
            }
         } else if (fActivations[direction] == "Softplus") {
            for (size_t i=hseq_start; i < hseq_end; i++) {
               hidden_state[i] = log(1. + exp(hidden_state[i]));
            }
         } else {
            throw
               std::runtime_error("TMVA - Invalid fActivations = {" + fActivations[direction] + "}");
         }
      }
   }

   // TODO Copy hidden_state into y and y_h

   if (bias) {
      delete[] bias;
   }
   if (fLayout == 1) {
      delete[] input;
      delete[] initial_hidden_state;
      if (Y.GetSize() ==0) delete[] hidden_state;
   }
   delete[] feedforward;
}


}
}
}

#endif
