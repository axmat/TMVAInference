#ifndef TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_CONV
#define TMVA_EXPERIMENTAL_SOFIE_ROPERATOR_CONV

#include <cstddef>
#include <vector>

#include <TMVA/RTensor.hxx>

#include "Blas.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

// Operator Conv
template <typename T> class ROperatorConv {
private:
   std::string fAutopad = "NOTSET";            // Auto padding
   std::vector<std::size_t> fDilations = {};   // Dilations
   std::size_t fGroup = 1;                     // Number of groups
   std::vector<std::size_t> fKernelShape = {}; // Kernel shape
   std::vector<std::size_t> fPads = {};        // Pads
   std::vector<std::size_t> fStrides = {};     // Strides

protected:
   /* Stack tensors column by column. */
   void Im2Col(const RTensor<T> &X, RTensor<T> &XCol,
               const std::size_t &kernelHeight,
               const std::size_t &kernelWidth,
               const std::size_t &stridesHeight,
               const std::size_t &stridesWidth);

public:
   /** Constructors. */
   ROperatorConv(const std::string &autopad,
                 const std::vector<std::size_t> &dilations,
                 const std::size_t &group,
                 const std::vector<std::size_t> &kernelShape,
                 const std::vector<std::size_t> &pads,
                 const std::vector<std::size_t> &strides)
       : fAutopad(autopad), fDilations(dilations), fGroup(group),
         fKernelShape(kernelShape), fPads(pads), fStrides(strides) {}

   /** Default constructor. */
   ROperatorConv() = default;

   /** Forward pass using blas. */
   void Forward_blas(const RTensor<T> &X /* Input */,
                     const RTensor<T> &W /* Weights */,
                     const RTensor<T> &B /* Bias */,
                           RTensor<T> &Y /* Output */);
};

template <typename T>
void ROperatorConv<T>::Im2Col(const RTensor<T> &X, RTensor<T> &XCol,
                              const std::size_t &kernelHeight,
                              const std::size_t &kernelWidth,
                              const std::size_t &stridesHeight,
                              const std::size_t &stridesWidth) {
   std::size_t batchSize = X.GetShape()[0];
   std::size_t channels  = X.GetShape()[1];
   std::size_t height    = X.GetShape()[2];
   std::size_t width     = X.GetShape()[3];

   std::size_t col = 0;

   for (std::size_t n = 0; n < batchSize; n++) {
      for (std::size_t c = 0; c < channels; c++) {
         for (std::size_t h = 0; h < height - kernelHeight + 1;
              h += stridesHeight) {
            for (std::size_t w = 0; w < width - kernelWidth + 1;
                 w += stridesWidth) {
               // Matrix of size kernelHeight x kernelWidth concatenated into
               // one long column
               std::size_t row = 0;
               for (std::size_t x = 0; x < kernelHeight; x++) {
                  for (std::size_t y = 0; y < kernelWidth; y++) {
                     XCol(row, col) = X(n, c, h + x, w + y);
                     row++;
                  }
               }
               col++;
            }
         }
      }
   }
}


template <typename T>
void ROperatorConv<T>::Forward_blas(const RTensor<T> &X,
                                    const RTensor<T> &W,
                                    const RTensor<T> &B,
                                          RTensor<T> &Y) {
   std::size_t inputSize = X.GetShape().size();
   if (inputSize == 4) { // 2d convolution
      // Input has size NxCxHxW
      std::size_t batchSize = X.GetShape()[0];
      std::size_t channels = X.GetShape()[1];
      std::size_t height = X.GetShape()[2];
      std::size_t width = X.GetShape()[3];

      // Number of groups
      std::size_t group = (fGroup > 0) ? fGroup : channels / W.GetShape()[1];
      if (group > 1)
         throw std::runtime_error("TMVA::SOFIE - Grouped 2d Convolution not implemented");

      // k Kernels, each kernelHeight x kernelWidth and C channels
      std::size_t kernels = W.GetShape()[0];
      std::size_t kHeight =
          (fKernelShape.empty()) ? W.GetShape()[2] : fKernelShape[0];
      std::size_t kWidth =
          (fKernelShape.empty()) ? W.GetShape()[3] : fKernelShape[1];

      // Dilations
      std::size_t dilationsHeight;
      std::size_t dilationsWidth;
      if (fDilations.empty()) {
         dilationsHeight = 1;
         dilationsWidth = 1;
      } else {
         dilationsHeight = fDilations[0];
         dilationsWidth = fDilations[1];
      }
      // kernels shape
      std::size_t kernelHeight = kHeight + (dilationsHeight - 1) * (kHeight - 1);
      std::size_t kernelWidth = kWidth + (dilationsWidth - 1) * (kWidth - 1);
      // Padding
      std::size_t padsHeightBegin;
      std::size_t padsHeightEnd;
      std::size_t padsWidthBegin;
      std::size_t padsWidthEnd;
      if (fAutopad == "NOTSET") { // Explicit padding
         if (fPads.empty()) {     // Default strides
            padsHeightBegin = 1;
            padsHeightEnd = 1;
            padsWidthBegin = 1;
            padsWidthEnd = 1;
         } else { // Stride along each spatial axis
            padsHeightBegin = fPads[0];
            padsHeightEnd = fPads[2];
            padsWidthBegin = fPads[1];
            padsWidthEnd = fPads[3];
         }
      } else if (fAutopad == "SAME_UPPER" || fAutopad == "SAME_LOWER") {
         padsHeightBegin = (kernelHeight - 1) / 2;
         padsHeightEnd = kernelHeight / 2;
         padsWidthBegin = (kernelWidth - 1) / 2;
         padsWidthEnd = kernelWidth / 2;
         if (kernelHeight % 2 == 1) {
            fAutopad == "SAME_UPPER" ? padsHeightEnd++ : padsHeightBegin++;
         }
         if (kernelWidth % 2 == 1) {
            fAutopad == "SAME_UPPER" ? padsWidthEnd++ : padsWidthBegin++;
         }
      } else if (fAutopad == "VALID") { // No padding
         padsHeightBegin = 0;
         padsHeightEnd = 0;
         padsWidthBegin = 0;
         padsWidthEnd = 0;
      } else {
         std::stringstream ss;
         ss << "Invalid padding fAutopad=";
         ss << fAutopad;
         throw std::runtime_error(ss.str());
      }

      // Strides
      std::size_t stridesHeight = 0;
      std::size_t stridesWidth = 0;
      if (fStrides.empty()) {
         stridesHeight = 1;
         stridesWidth = 1;
      } else {
         stridesHeight = fStrides[0];
         stridesWidth = fStrides[1];
      }

      RTensor<T> XPad(
          {batchSize, channels, height + padsHeightBegin + padsHeightEnd,
           width + padsWidthBegin + padsWidthEnd},
          {channels * (height + padsHeightBegin + padsHeightEnd) *
               (width + padsWidthBegin + padsWidthEnd),
           (height + padsHeightBegin + padsHeightEnd) *
               (width + padsWidthBegin + padsWidthEnd),
           width + padsWidthBegin + padsWidthEnd, 1});
      // Padding the input with zeros
      for (std::size_t n = 0; n < batchSize; n++) {
         for (std::size_t c = 0; c < channels; c++) {
            for (std::size_t h = 0; h < height; h++) {
               for (std::size_t w = 0; w < width; w++) {
                  XPad(n, c, h + padsHeightBegin, w + padsWidthBegin) =
                      X(n, c, h, w);
               }
            }
         }
      }
      // Output shape
      std::size_t outputHeight = (height + padsHeightBegin + padsHeightEnd - kernelHeight
             + stridesHeight) / stridesHeight;
      std::size_t outputWidth = (width + padsWidthBegin + padsWidthEnd - kernelWidth
             + stridesWidth) / stridesWidth;
      RTensor<T> XCol({channels * kernelHeight * kernelWidth, batchSize * outputHeight * outputWidth},
                      {1, channels * kernelHeight * kernelWidth},
                      MemoryLayout::ColumnMajor);
      // Unroll the input tensor
      Im2Col(XPad, XCol, kernelHeight, kernelWidth, stridesHeight,
             stridesWidth);
      // Convolution kernels,  kernels x channels x kernelHeight x KernelWidth
      RTensor<T> F({kernels, channels * kernelHeight * kernelWidth},
                   {1, kernels},
                   MemoryLayout::ColumnMajor);
      // Vectorize the (dilated)convolution kernels into a matrix
      for (std::size_t k = 0; k < kernels; k++) {
         for (std::size_t c = 0; c < channels; c++) {
            for (std::size_t h = 0; h < kHeight; h++) {
               for (std::size_t w = 0; w < kWidth; w++) {
                  F(k, c * kernelHeight * kernelWidth + h * dilationsHeight * kernelWidth 
                   + w * dilationsWidth) = W(k, c, h, w);
               }
            }
         }
      }
      // Compute the output, vec(Y) = vec(F * XCol) of size
      // kernels x batchSize x outputHeight x outputWidth
      char transa = 'N';
      char transb = 'N';
      int m = F.GetShape()[0];
      int n = XCol.GetShape()[1];
      int k = F.GetShape()[1];
      T alpha = 1.0;
      T beta = 0.0;
      int lda = m;
      int ldb = k;

      Blas::Gemm<T>(&transa, &transb, &m, &n, &k, &alpha, F.GetData(),
                    &lda, XCol.GetData(), &ldb, &beta, Y.GetData(), &m);
      // Add bias
      for(std::size_t k=0; k < kernels; k++) {
         for(std::size_t n=0; n < batchSize; n++) {
            for(std::size_t h=0;h < outputHeight; h++) {
               for(std::size_t w=0; w < outputWidth; w++) {
                  Y(k, n, h, w) += B(k);
               }
            }
         }
      }
   } else {
      std::stringstream ss;
      ss << "TMVA::SOFIE - Convolution not implemented for input size =";
      ss << inputSize;
      throw std::runtime_error(ss.str());
   }
}


} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
