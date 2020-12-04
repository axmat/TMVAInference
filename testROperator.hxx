#ifndef TEST_ROPERATOR
#define TEST_ROPERATOR

#include <cstddef>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

#include <TMVA/RTensor.hxx>

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
      if (abs(static_cast<double>(dataA[i] - dataB[i])) > tol)
         return false;

   return true;
}

#endif
