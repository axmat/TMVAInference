#include <iostream>
#include <vector>
#include <numeric>

template<typename T>
std::vector<T> Broadcast(const T* data, const std::vector<size_t>& shape, const std::vector<size_t>& targetShape) {
   // Size of the shapes
   size_t size = shape.size();
   // Current length of the broadcasted tensor
   size_t curLength = 1;
   for (size_t i = 0; i < size; i++) {
      curLength *= shape[i];
   }
   // newShape is a vector of vectors of size equal to the number of the current dimension
   // along which we are broadcasting the tensor
   std::vector<T> broadcastedData(data, data + curLength);
   // Size of the previous dimension
   size_t prevDim = 1;

   for (size_t idx = 0; idx < size; idx++) {
      size_t dim = shape[idx];
      size_t targetDim = targetShape[idx];
      if (dim == 1 && targetDim > 1) {
         // Set the new length of the data
         size_t newLength = curLength * targetDim;
         // New
         std::vector<T> newData(newLength);
         // View the data as a list of prevDim arrays of size arrayLength
         size_t arrayLength = curLength / prevDim;
         // New array length
         //size_t newArrayLength = arrayLength * targetDim;
         std::cout << "previous dim = " << prevDim << std::endl;
         std::cout << "array length = " << arrayLength << std::endl;
         // Broadcast each array dim times
         if (arrayLength > 1) {
            // If each array has at least two elements
            for (size_t prevIdx = 0; prevIdx < prevDim; prevIdx++) {
               for (size_t targetIdx = 0; targetIdx < targetDim; targetIdx++) {
                  size_t offset = prevIdx * arrayLength * targetDim + targetIdx * arrayLength;
                  std::copy(broadcastedData.begin() + prevIdx * arrayLength,
                     broadcastedData.begin() + (prevIdx + 1) * arrayLength,
                     newData.begin() + offset);
               }
            }
         } else {
            for (size_t prevIdx = 0; prevIdx < prevDim; prevIdx++) {
               std::fill(newData.begin() + prevIdx * targetDim,
                  newData.begin() + (prevIdx + 1) * targetDim, broadcastedData[prevIdx]);
            }
         }
         // Update current length
         curLength = newLength;
         // Update broadcasted data
         for (size_t i = 0; i < newData.size(); i++) std::cout << newData[i] << " ";
         std::cout << std::endl;
         broadcastedData = std::vector<T>(newData);
      }
      // Update previous dim
      prevDim = dim;
   }
   return broadcastedData;
}

int main() {
   {
      size_t length = 3 * 2;
      std::vector<float> x(3);
      std::iota(x.begin(), x.end(), 0.);
      auto z = Broadcast(x.data(), {3, 1}, {3, 2});
      for (size_t i = 0; i < length; i++) {
         std::cout << z[i] << " ";
      }
      std::cout << "\n";
   }
   {
      size_t length = 2 * 4;
      std::vector<float> x(2);
      std::iota(x.begin(), x.end(), 0.);
      auto z = Broadcast(x.data(), {2, 1}, {2, 4});
      for (size_t i = 0; i < length; i++) {
         std::cout << z[i] << " ";
      }
      std::cout << "\n";
   }

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
      std::cout << "{2, 1, 3, 1, 2} to {2, 2, 3, 2, 2}\n";
      size_t length = 2 * 3 * 2;
      std::vector<float> x(2 * 3 * 2);
      std::iota(x.begin(), x.end(), 0.);
      auto z = Broadcast(x.data(), {2, 1, 3, 1, 2}, {2, 2, 3, 2, 2});
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
