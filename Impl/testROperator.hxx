#ifndef TEST_ROPERATOR
#define TEST_ROPERATOR

#include <cstddef>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

#include <TMVA/RTensor.hxx>
#include <stdexcept>

// Approximately compare two tensors
template<typename T>
bool IsApprox(const TMVA::Experimental::RTensor<T> &A,
              const TMVA::Experimental::RTensor<T> &B, double tol) {

   const auto &shapeA = A.GetShape();
   std::size_t ndims = shapeA.size();

   if (ndims != B.GetShape().size())
      return false;

   const auto &shapeB = B.GetShape();
   for(std::size_t i=0; i < ndims; i++)
      if(shapeA[i] != shapeB[i])
         return false;

   std::size_t size = A.GetSize();
   const auto &dataA = A.GetData();
   const auto &dataB = B.GetData();
   for (std::size_t i=0; i < size; i++)
      if (std::abs(static_cast<double>(dataA[i] - dataB[i])) > tol)
         return false;

   return true;
}

// Print
template<typename T>
void Print(T* A, std::vector<size_t> shape, const std::string& name="") {
   std::cout << name << std::endl;
   size_t size = shape.size();
   std::cout << "[";
   if (size == 1) {
      for (size_t i=0; i<shape[0]; i++)
         std::cout << A[i] << " ";
      std::cout << "]\n";
   } else if (size == 2) {
      for (size_t h=0; h<shape[0]; h++) {
         std::cout << "[";
         for(size_t w=0; w<shape[1]; w++) {
            std::cout << A[h * shape[1] + w];
            if (w < shape[1]) std::cout << " ";
         }
         std::cout << "]";
         if (h < shape[0] - 1) std::cout << "\n";
      }
      std::cout << "]\n";
   } else if (size == 3) {
      for (size_t c=0; c<shape[0]; c++) {
         if (c>0) std::cout << " ";
         std::cout << "[";
         for (size_t h=0; h<shape[1]; h++) {
            if (h > 0) std::cout << "  ";
            std::cout << "[";
            for(size_t w=0; w<shape[2]; w++) {
               std::cout << A[c * shape[1] * shape[2] + h * shape[2] + w];
               if (w < shape[2]) std::cout << " ";
            }
            std::cout << "]";
            if (h < shape[1] - 1) std::cout << "\n";
         }
         std::cout << "]";
         if (c < shape[0] - 1) std::cout << "\n";
      }
      std::cout << "]\n";
   } else {
      throw
         std::runtime_error("TMVAInference - Print not Implemented for tensor of size " + std::to_string(size));
   }
}

#endif
