#include <iostream>
#include <vector>
#include <numeric>

template<typename T>
T* Broadcast(const T* data, const std::vector<size_t>& shape, const std::vector<size_t>& targetShape) {
   size_t targetSize = targetShape.size();
   //size_t targetLength = ConvertShapeToLength(targetShape);
   //T* newData = new T[targetLength];
   size_t prevLength = 1;
   for (size_t i = 0; i < shape.size(); i++) prevLength *= shape[i];
   std::vector<T> prevData(data, data + prevLength);
   // Stride
   size_t prevStride = 1;
   for (size_t i = 1; i < shape.size(); i++) {
      prevStride *= shape[i];
   }
   //std::vector<T> newData(prevData);
   T* newData = new T[prevLength];
   std::cout << "First length = " << prevLength << std::endl;
   // The initial previous shape is the original data viewed as a contiguous arrays of size 
   // previousDim*stride concatenated into one long row
   for (size_t dim = 1; dim < targetSize - 1; dim++) {
      // Previous dimension
      size_t prevDim = targetShape[dim - 1];
      // Target dimension
      size_t newDim = targetShape[dim];
      if (shape[dim] == 1) {
         std::cout << "\nDo broadcasting in index = " << dim << std::endl;
         size_t newLength = prevLength;// * newDim;
         if (shape[dim] == 1) newLength *= newDim;
         std::cout << "NewLength = " << newLength << std::endl;
         // Broadcast for the dim-1th dimension
         // Repeat each subarray of size stride newDim times
         //newData.resize(newLength);
         delete[] newData;
         newData = new T[newLength];
         size_t newStride = prevStride;
         if (shape[dim] == 1) newStride *= newDim;
         std::cout << "NewStride = " << newStride << std::endl;
         for (size_t prevIdx = 0; prevIdx < prevDim; prevIdx++) {
            //size_t prevOffset = prevIdx * prevStride;
            for (size_t newIdx = 0; newIdx < newDim; newIdx++) {
               size_t newOffset = prevIdx * newStride + newIdx * prevStride;
               std::copy(prevData.begin() + prevIdx * prevStride,
                  prevData.begin() + (prevIdx + 1) * prevStride,
                  newData + newOffset);
            }
         }
         // Update prevLength
         prevLength = newLength;
         // Update prevStride
         prevStride = newStride;
         // Update prevData
         prevData = std::vector<T>(newData, newData + newLength);
         //prevData.resize(newLength);
         //std::copy(newData, newData + newLength, prevData.begin());
         for (size_t i = 0; i < newLength; i++) std::cout << prevData[i] << " ";
         std::cout << std::endl;
      }
   }
   return newData;
}

int main() {
   {
      size_t length = 2 * 3 * 4;
      std::vector<float> x(2 * 4);
      std::iota(x.begin(), x.end(), 0.);
      auto z = Broadcast(x.data(), {2, 1, 4}, {2, 3, 4});
      for (size_t i = 0; i < length; i++) {
         std::cout << z[i] << " ";
      }
      std::cout << "\n";
   }

   {
      std::cout << std::endl;
      size_t length = 2 * 3 * 4 * 2 * 3;
      std::vector<float> x(2 * 4 * 3);
      std::iota(x.begin(), x.end(), 0.);
      auto z = Broadcast(x.data(), {2, 1, 4, 1, 3}, {2, 3, 4, 2, 3});
      for (size_t i = 0; i < length; i++) {
         std::cout << z[i] << " ";
      }
      std::cout << "\n";
   }

   return 0;
}
