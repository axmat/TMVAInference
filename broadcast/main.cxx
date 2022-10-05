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
   // newShape is an aray of size equal to dimension along which we are broadcasting the tensor
   std::vector<T> broadcastedData(data, data + curLength);
   // Product of the previous dimensions of targetShape
   size_t arrayNum = 1;

   for (size_t idx = 0; idx < size; idx++) {
      size_t dim = shape[idx];
      size_t targetDim = targetShape[idx];
      if (dim == 1 && targetDim > 1) {
         // Set the new length of the data
         size_t newLength = curLength * targetDim;
         // New broadcasted data
         std::vector<T> newData(newLength);
         // View the data as a list of arrayNum arrays of size arrayLength
         size_t arrayLength = curLength / arrayNum;
         // Broadcast each array dim times
         if (arrayLength > 1) {
            // If each array has at least two elements
            for (size_t arrayIdx = 0; arrayIdx < arrayNum; arrayIdx++) {
               for (size_t targetIdx = 0; targetIdx < targetDim; targetIdx++) {
                  size_t offset = arrayIdx * arrayLength * targetDim + targetIdx * arrayLength;
                  std::copy(broadcastedData.begin() + arrayIdx * arrayLength,
                     broadcastedData.begin() + (arrayIdx + 1) * arrayLength,
                     newData.begin() + offset);
               }
            }
         } else {
            for (size_t arrayIdx = 0; arrayIdx < arrayNum; arrayIdx++) {
               std::fill(newData.begin() + arrayIdx * targetDim,
                  newData.begin() + (arrayIdx + 1) * targetDim, broadcastedData[arrayIdx]);
            }
         }
         // Update current length
         curLength = newLength;
         // Update broadcasted data
         //for (size_t i = 0; i < newData.size(); i++) std::cout << newData[i] << " ";
         broadcastedData = std::vector<T>(newData);
      }
      // Update previous dim
      //prevDim = dim;
      // Update k
      arrayNum *= targetDim;
   }
   return broadcastedData;
}

int main() {
   {
      std::cout << "\n{3, 1} to {3, 2}\n";
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
      std::cout << "\n{2, 1} to {2, 4}\n";
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
      std::cout << "\n{2, 1, 4} to {2, 3, 4}\n";
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
      std::cout << "\n{2, 1, 3, 1, 2} to {2, 2, 3, 2, 2}\n";
      size_t length = 2 * 2 * 3 * 2 * 2;
      std::vector<float> x(2 * 3 * 2);
      std::iota(x.begin(), x.end(), 0.);
      auto z = Broadcast(x.data(), {2, 1, 3, 1, 2}, {2, 2, 3, 2, 2});
      for (size_t i = 0; i < length; i++) {
         std::cout << z[i] << " ";
      }
      std::cout << "\n";
   }


   {
      std::cout << "{2, 1, 4, 1, 3} to {2, 3, 4, 2, 3}\n";
      std::cout << std::endl;
      size_t length = 2 * 3 * 4 * 2 * 3;
      std::vector<float> x(2 * 4 * 3);
      std::iota(x.begin(), x.end(), 0.);
      auto z = Broadcast(x.data(), {2, 1, 4, 1, 3}, {2, 3, 4, 2, 3});
      for (size_t i = 0; i < length; i++) {
         std::cout << z[i] << " ";
      }
      std::cout << "\n";

      float out[] = {
         0.,  1.,  2.,  0.,  1.,  2.,  3.,  4.,  5.,  3.,  4.,  5.,  6.,
        7.,  8.,  6.,  7.,  8.,  9., 10., 11.,  9., 10., 11.,  0.,  1.,
        2.,  0.,  1.,  2.,  3.,  4.,  5.,  3.,  4.,  5.,  6.,  7.,  8.,
        6.,  7.,  8.,  9., 10., 11.,  9., 10., 11.,  0.,  1.,  2.,  0.,
        1.,  2.,  3.,  4.,  5.,  3.,  4.,  5.,  6.,  7.,  8.,  6.,  7.,
        8.,  9., 10., 11.,  9., 10., 11., 12., 13., 14., 12., 13., 14.,
       15., 16., 17., 15., 16., 17., 18., 19., 20., 18., 19., 20., 21.,
       22., 23., 21., 22., 23., 12., 13., 14., 12., 13., 14., 15., 16.,
       17., 15., 16., 17., 18., 19., 20., 18., 19., 20., 21., 22., 23.,
       21., 22., 23., 12., 13., 14., 12., 13., 14., 15., 16., 17., 15.,
       16., 17., 18., 19., 20., 18., 19., 20., 21., 22., 23., 21., 22.,
       23.};
      for (size_t i = 0; i < length; i++) {
         if (z[i] != out[i]) {
            std::cout << "\nz != out\n";
            break;
         }
      }
   }

   return 0;
}
