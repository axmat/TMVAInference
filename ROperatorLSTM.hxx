#ifndef TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_LSTM
#define TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_LSTM

#include <TMVA/RTensor.hxx>
#include <iostream>
#include <stdexcept>

//#include "Blas.hxx"

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
   void Forward_blas(const RTensor<T> &X,
                     const RTensor<T> &W,
                           RTensor<T> &R,
                           RTensor<T> &B,
                           RTensor<T> &SequenceLens,
                           RTensor<T> &InitialH,
                           RTensor<T> &InitialC,
                           RTensor<T> &P,
                           RTensor<T> &Y,
                           RTensor<T> &Yh,
                           RTensor<T> &Yc);

};

template<typename T>
void ROperatorLSTM<T>::Forward_blas(const RTensor<T> &X,
                                    const RTensor<T> &W,
                                          RTensor<T> &R,
                                          RTensor<T> &B,
                                          RTensor<T> &SequenceLens,
                                          RTensor<T> &InitialH,
                                          RTensor<T> &InitialC,
                                          RTensor<T> &P,
                                          RTensor<T> &Y,
                                          RTensor<T> &Yh,
                                          RTensor<T> &Yc) {
   // Activations
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
   // The input X has shape
   //    seqLength x batchSize x inputSize if fLayout=0
   //    batchSize x seqLength x inputSize if fLayout=1
   size_t seqLength;
   size_t batchSize;
   size_t inputSize;
   if (fLayout == 0) {
      seqLength = X.GetShape()[0];
      batchSize = X.GetShape()[1];
      inputSize = X.GetShape()[2];
   } else if (fLayout == 1) {
      batchSize = X.GetShape()[0];
      seqLength = X.GetShape()[1];
      inputSize = X.GetShape()[2];
   } else {
      throw
         std::runtime_error("TMVA SOFIE - Invalid fLayout");
   }

   std::size_t numDirections = W.GetShape()[0];
   std::size_t hiddenSize = R.GetShape()[2];

   if (B.GetSize() == 0) {
      B = RTensor<T>({numDirections, 8 * hiddenSize});
   }

   if (SequenceLens.GetSize() == 0) {
      SequenceLens = RTensor<T>({batchSize});
      std::fill(SequenceLens.begin(), SequenceLens.end(), seqLength);
   }

   if (InitialH.GetSize() == 0) {
      if (fLayout == 0) {
         InitialH = RTensor<T>({numDirections, batchSize, hiddenSize});
      } else if (fLayout == 1) {
         InitialH = RTensor<T>({batchSize, numDirections, hiddenSize});
      } else {
         throw
            std::runtime_error("TMVA SOFIE - Invalid fLayout");
      }
   }

   if (InitialC.GetSize() == 0) {
      if (fLayout == 0) {
         InitialC = RTensor<T>({numDirections, batchSize, hiddenSize});
      } else if (fLayout == 1) {
         InitialC = RTensor<T>({batchSize, numDirections, hiddenSize});
      } else {
         throw
            std::runtime_error("TMVA SOFIE - Invalid fLayout");
      }
   }

   if (P.GetSize() == 0) {
      P = RTensor<T>({numDirections, 3*hiddenSize});
   }

   if (fDirection == "forward") {
      // TODO Forward LSTM
   } else if (fDirection == "reverse") {
      // TODO Reverse LSTM
   } else if (fDirection == "bidirectional") {
      // TODO Bidirectional LSTM
   }
}


}
}
}

#endif
