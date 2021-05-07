#ifndef TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_GRU
#define TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_GRU

#include <TMVA/RTensor.hxx>
#include <iostream>

//#include "Blas.hxx"

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
                     RTensor<T> &SequenceLens,
                     RTensor<T> &InitialH,
                     RTensor<T> &Y,
                     RTensor<T> &Yh);

};

template<typename T>
void ROperatorGRU<T>::Forward_blas(RTensor<T> &X,
                                   RTensor<T> &W,
                                   RTensor<T> &R,
                                   RTensor<T> &B,
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
   // shape of the input X
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
   size_t numDirections = W.GetShape()[0];
   size_t hiddenSize = W.GetShape()[1];
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

   if (fDirection == "forward") {
      // TODO Forward GRU
   } else if (fDirection == "reverse") {
      // TODO Reverse GRU
   } else if (fDirection == "bidirectional") {
      // TODO Bidirectional GRU
   }
}


}
}
}

#endif
