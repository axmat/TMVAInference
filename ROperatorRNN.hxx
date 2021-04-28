#ifndef TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_RNN
#define TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_RNN

#include <TMVA/RTensor.hxx>
#include <iostream>

//#include "Blas.hxx"

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
   void Forward_blas(const RTensor<T> &X,
                     const RTensor<T> &W,
                           RTensor<T> &B,
                           RTensor<T> &R,
                           RTensor<T> &SequenceLens,
                           RTensor<T> &InitialH,
                           RTensor<T> &Y,
                           RTensor<T> &Yh);

};

template<typename T>
void ROperatorRNN<T>::Forward_blas(const RTensor<T> &X,
                                   const RTensor<T> &W,
                                         RTensor<T> &B,
                                         RTensor<T> &R,
                                         RTensor<T> &SequenceLens,
                                         RTensor<T> &InitialH,
                                         RTensor<T> &Y,
                                         RTensor<T> &Yh) {
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
   // W of shape numDirections x hiddenSize x inputSize
   std::size_t numDirections = W.GetShape()[0];
   std::size_t hiddenSize = W.GetShape()[1];
   // R has shape numDirections x hiddenSize x hiddenSize
   // B [optional] has shape numDirections x 2*hiddenSize
   if (B.GetSize() == 0) {
      B = RTensor<T>({numDirections, 2 * hiddenSize});
   }
   // SequenceLens [optional] of shape batchSize
   if (SequenceLens.GetSize() == 0) {
      SequenceLens = RTensor<T>({batchSize});
      std::fill(SequenceLens.begin(), SequenceLens.end(), seqLength);
   }
   // InitalH [optional]
   if (InitialH.GetSize() == 0) {
      InitialH = RTensor<T>({numDirections, batchSize, hiddenSize});
   }
   // Y has shape
   //    seqLength x numDirections x batchSize x hiddenSize if fLayout=0
   //    batchSize x seqLength x numDirections x hiddenSize if fLayout=1
   // Yh has shape
   //    numDirections x batchSize x hiddenSize if fLayout=0
   //    batchSize x numDirections x hiddenSize if fLayout=1

   if (fDirection == "forward") {
      // TODO Forward RNN
   } else if (fDirection == "reverse") {
      // TODO Reverse RNN
   } else if (fDirection == "bidirectional") {
      // TODO Bidirectional RNN
   }
}


}
}
}

#endif
