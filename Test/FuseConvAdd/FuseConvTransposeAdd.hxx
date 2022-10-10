//Code generated automatically by TMVA for Inference of Model file [Conv2DTranspose_Relu_Sigmoid.onnx] at [Thu Sep 22 20:16:09 2022] 

#ifndef TMVA_SOFIE_CONV2DTRANSPOSE_RELU_SIGMOID
#define TMVA_SOFIE_CONV2DTRANSPOSE_RELU_SIGMOID

#include<algorithm>
#include<vector>
#include<cmath>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_Conv2DTranspose_Relu_Sigmoid{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
struct Session {
std::vector<float> fTensor_StatefulPartitionedCalldecoderconv2dtransposeBiasAddReadVariableOp0 = std::vector<float>(180);
float * tensor_StatefulPartitionedCalldecoderconv2dtransposeBiasAddReadVariableOp0 = fTensor_StatefulPartitionedCalldecoderconv2dtransposeBiasAddReadVariableOp0.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderconv2dtransposeconv2dtransposeReadVariableOp0 = std::vector<float>(291600);
float * tensor_StatefulPartitionedCalldecoderconv2dtransposeconv2dtransposeReadVariableOp0 = fTensor_StatefulPartitionedCalldecoderconv2dtransposeconv2dtransposeReadVariableOp0.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderconv2dtranspose1conv2dtransposeReadVariableOp0 = std::vector<float>(145800);
float * tensor_StatefulPartitionedCalldecoderconv2dtranspose1conv2dtransposeReadVariableOp0 = fTensor_StatefulPartitionedCalldecoderconv2dtranspose1conv2dtransposeReadVariableOp0.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderconv2dtranspose2conv2dtransposeReadVariableOp0 = std::vector<float>(64800);
float * tensor_StatefulPartitionedCalldecoderconv2dtranspose2conv2dtransposeReadVariableOp0 = fTensor_StatefulPartitionedCalldecoderconv2dtranspose2conv2dtransposeReadVariableOp0.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderdense1BiasAddReadVariableOp0 = std::vector<float>(5940);
float * tensor_StatefulPartitionedCalldecoderdense1BiasAddReadVariableOp0 = fTensor_StatefulPartitionedCalldecoderdense1BiasAddReadVariableOp0.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderdense1MatMulReadVariableOp0 = std::vector<float>(89100);
float * tensor_StatefulPartitionedCalldecoderdense1MatMulReadVariableOp0 = fTensor_StatefulPartitionedCalldecoderdense1MatMulReadVariableOp0.data();
std::vector<float> fTensor_constfoldopt118 = std::vector<float>(40500);
float * tensor_constfoldopt118 = fTensor_constfoldopt118.data();
std::vector<float> fTensor_constfoldopt121 = std::vector<float>(63450);
float * tensor_constfoldopt121 = fTensor_constfoldopt121.data();
std::vector<float> fTensor_conv2dtranspose2 = std::vector<float>(40500);
float * tensor_conv2dtranspose2 = fTensor_conv2dtranspose2.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderdense1BiasAdd0 = std::vector<float>(5940);
float * tensor_StatefulPartitionedCalldecoderdense1BiasAdd0 = fTensor_StatefulPartitionedCalldecoderdense1BiasAdd0.data();
std::vector<float> fTensor_ConvTranspose790 = std::vector<float>(28980);
float * tensor_ConvTranspose790 = fTensor_ConvTranspose790.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderconv2dtransposeRelu0 = std::vector<float>(28980);
float * tensor_StatefulPartitionedCalldecoderconv2dtransposeRelu0 = fTensor_StatefulPartitionedCalldecoderconv2dtransposeRelu0.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderconv2dtranspose1conv2dtranspose0 = std::vector<float>(63450);
float * tensor_StatefulPartitionedCalldecoderconv2dtranspose1conv2dtranspose0 = fTensor_StatefulPartitionedCalldecoderconv2dtranspose1conv2dtranspose0.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderdense1Relu0 = std::vector<float>(5940);
float * tensor_StatefulPartitionedCalldecoderdense1Relu0 = fTensor_StatefulPartitionedCalldecoderdense1Relu0.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderreshapeReshape0 = std::vector<float>(5940);
float * tensor_StatefulPartitionedCalldecoderreshapeReshape0 = fTensor_StatefulPartitionedCalldecoderreshapeReshape0.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderconv2dtransposeconv2dtranspose370 = std::vector<float>(5940);
float * tensor_StatefulPartitionedCalldecoderconv2dtransposeconv2dtranspose370 = fTensor_StatefulPartitionedCalldecoderconv2dtransposeconv2dtranspose370.data();
std::vector<float> fTensor_BroadcastedStatefulPartitionedCalldecoderconv2dtransposeBiasAddReadVariableOp0 = std::vector<float>(28980);
float * tensor_BroadcastedStatefulPartitionedCalldecoderconv2dtransposeBiasAddReadVariableOp0 = fTensor_BroadcastedStatefulPartitionedCalldecoderconv2dtransposeBiasAddReadVariableOp0.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderconv2dtranspose2BiasAdd0 = std::vector<float>(40500);
float * tensor_StatefulPartitionedCalldecoderconv2dtranspose2BiasAdd0 = fTensor_StatefulPartitionedCalldecoderconv2dtranspose2BiasAdd0.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderconv2dtranspose1BiasAdd0 = std::vector<float>(63450);
float * tensor_StatefulPartitionedCalldecoderconv2dtranspose1BiasAdd0 = fTensor_StatefulPartitionedCalldecoderconv2dtranspose1BiasAdd0.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderconv2dtranspose1Relu0 = std::vector<float>(63450);
float * tensor_StatefulPartitionedCalldecoderconv2dtranspose1Relu0 = fTensor_StatefulPartitionedCalldecoderconv2dtranspose1Relu0.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderconv2dtranspose2conv2dtranspose0 = std::vector<float>(40500);
float * tensor_StatefulPartitionedCalldecoderconv2dtranspose2conv2dtranspose0 = fTensor_StatefulPartitionedCalldecoderconv2dtranspose2conv2dtranspose0.data();
std::vector<float> fTensor_StatefulPartitionedCalldecoderconv2dtranspose2Sigmoid0 = std::vector<float>(40500);
float * tensor_StatefulPartitionedCalldecoderconv2dtranspose2Sigmoid0 = fTensor_StatefulPartitionedCalldecoderconv2dtranspose2Sigmoid0.data();

std::vector<float> fVec_op_4_f = std::vector<float>(291600);
std::vector<float> fVec_op_4_xcol = std::vector<float>(1449);

std::vector<float> fVec_op_6_f = std::vector<float>(145800);
std::vector<float> fVec_op_6_xcol = std::vector<float>(6345);

std::vector<float> fVec_op_9_f = std::vector<float>(64800);
std::vector<float> fVec_op_9_xcol = std::vector<float>(14400);


Session(std::string filename ="") {
   if (filename.empty()) filename = "Conv2DTranspose_Relu_Sigmoid.dat";
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()){
      throw std::runtime_error("tmva-sofie failed to open file for input weights");
   }
   std::string tensor_name;
   int length;
   f >> tensor_name >> length;
   if (tensor_name != "tensor_StatefulPartitionedCalldecoderconv2dtransposeBiasAddReadVariableOp0" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_StatefulPartitionedCalldecoderconv2dtransposeBiasAddReadVariableOp0 , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 180) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 180 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
    for (int i =0; i < length; ++i) 
       f >> tensor_StatefulPartitionedCalldecoderconv2dtransposeBiasAddReadVariableOp0[i];
   f >> tensor_name >> length;
   if (tensor_name != "tensor_StatefulPartitionedCalldecoderconv2dtransposeconv2dtransposeReadVariableOp0" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_StatefulPartitionedCalldecoderconv2dtransposeconv2dtransposeReadVariableOp0 , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 291600) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 291600 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
    for (int i =0; i < length; ++i) 
       f >> tensor_StatefulPartitionedCalldecoderconv2dtransposeconv2dtransposeReadVariableOp0[i];
   f >> tensor_name >> length;
   if (tensor_name != "tensor_StatefulPartitionedCalldecoderconv2dtranspose1conv2dtransposeReadVariableOp0" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_StatefulPartitionedCalldecoderconv2dtranspose1conv2dtransposeReadVariableOp0 , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 145800) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 145800 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
    for (int i =0; i < length; ++i) 
       f >> tensor_StatefulPartitionedCalldecoderconv2dtranspose1conv2dtransposeReadVariableOp0[i];
   f >> tensor_name >> length;
   if (tensor_name != "tensor_StatefulPartitionedCalldecoderconv2dtranspose2conv2dtransposeReadVariableOp0" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_StatefulPartitionedCalldecoderconv2dtranspose2conv2dtransposeReadVariableOp0 , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 64800) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 64800 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
    for (int i =0; i < length; ++i) 
       f >> tensor_StatefulPartitionedCalldecoderconv2dtranspose2conv2dtransposeReadVariableOp0[i];
   f >> tensor_name >> length;
   if (tensor_name != "tensor_StatefulPartitionedCalldecoderdense1BiasAddReadVariableOp0" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_StatefulPartitionedCalldecoderdense1BiasAddReadVariableOp0 , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 5940) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 5940 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
    for (int i =0; i < length; ++i) 
       f >> tensor_StatefulPartitionedCalldecoderdense1BiasAddReadVariableOp0[i];
   f >> tensor_name >> length;
   if (tensor_name != "tensor_StatefulPartitionedCalldecoderdense1MatMulReadVariableOp0" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_StatefulPartitionedCalldecoderdense1MatMulReadVariableOp0 , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 89100) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 89100 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
    for (int i =0; i < length; ++i) 
       f >> tensor_StatefulPartitionedCalldecoderdense1MatMulReadVariableOp0[i];
   f >> tensor_name >> length;
   if (tensor_name != "tensor_constfoldopt118" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_constfoldopt118 , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 40500) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 40500 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
    for (int i =0; i < length; ++i) 
       f >> tensor_constfoldopt118[i];
   f >> tensor_name >> length;
   if (tensor_name != "tensor_constfoldopt121" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_constfoldopt121 , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 63450) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 63450 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
    for (int i =0; i < length; ++i) 
       f >> tensor_constfoldopt121[i];
   f.close();
   {
      std::vector<size_t> oldShape = { 180 };
      std::vector<size_t> newShape = { 180, 7, 23};
      oldShape.resize(newShape.size(), 1.);
      float * newData_ptr = TMVA::Experimental::SOFIE::UTILITY::Unidirectional_broadcast<float>(tensor_StatefulPartitionedCalldecoderconv2dtransposeBiasAddReadVariableOp0, oldShape, newShape);
      for (int i = 0; i < 1 ; i++)
         std::copy(newData_ptr, newData_ptr + 28980, tensor_BroadcastedStatefulPartitionedCalldecoderconv2dtransposeBiasAddReadVariableOp0 + i * 28980);
      delete [] newData_ptr;
   }
}

std::vector<float> infer(float* tensor_input5){

//--------- Gemm
   char op_0_transA = 'n';
   char op_0_transB = 'n';
   int op_0_m = 1;
   int op_0_n = 5940;
   int op_0_k = 15;
   float op_0_alpha = 1;
   float op_0_beta = 1;
   int op_0_lda = 15;
   int op_0_ldb = 5940;
   BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_StatefulPartitionedCalldecoderdense1MatMulReadVariableOp0, &op_0_ldb, tensor_input5, &op_0_lda, &op_0_beta, tensor_StatefulPartitionedCalldecoderdense1BiasAdd0, &op_0_n);

//------ RELU
   for (int id = 0; id < 5940 ; id++){
      tensor_StatefulPartitionedCalldecoderdense1Relu0[id] = ((tensor_StatefulPartitionedCalldecoderdense1BiasAdd0[id] > 0 )? tensor_StatefulPartitionedCalldecoderdense1BiasAdd0[id] : 0);
   }
   ///--------Reshape operator

   std::copy( fTensor_StatefulPartitionedCalldecoderdense1Relu0.begin(), fTensor_StatefulPartitionedCalldecoderdense1Relu0.end(), fTensor_StatefulPartitionedCalldecoderreshapeReshape0.begin() );
   ///------- Transpose operator

   for (size_t id = 0; id < 5940 ; id++){
      tensor_StatefulPartitionedCalldecoderconv2dtransposeconv2dtranspose370[id] = tensor_StatefulPartitionedCalldecoderreshapeReshape0[ ( id / 5940 ) * 5940 + ( (id % 33) / 11 ) * 1980 + ( (id % 11) ) * 180 + ( (id % 5940) / 33 )];
   }

//----  operator ConvTranspose op_4
   float * op_4_f = fVec_op_4_f.data();
   for (std::size_t oc = 0; oc < 180; oc++) {
      for (std::size_t ic = 0; ic < 180; ic++) {
         for (std::size_t kh = 0; kh < 3; kh++) {
            for (std::size_t kw = 0; kw < 3; kw++) {
               op_4_f[oc * 1620 + ic * 9 + kh * 3 + kw * 1  ] = tensor_StatefulPartitionedCalldecoderconv2dtransposeconv2dtransposeReadVariableOp0[oc * 1620 + ic * 9 + kh * 3 + kw ];
            }
         }
      }
   }
   char op_4_transA = 'N';
   char op_4_transB = 'N';
   int op_4_m = 161;
   int op_4_n = 180;
   int op_4_k = 9;
   float op_4_alpha = 1.0;
   float op_4_beta = 0.0;
   float * op_4_xcol = fVec_op_4_xcol.data();
   for (size_t n = 0; n < 1; n++) {
      size_t x_offset = n * 5940;
      size_t out_offset = n * 28980;
      TMVA::Experimental::SOFIE::UTILITY::Im2col<float>(tensor_StatefulPartitionedCalldecoderconv2dtransposeconv2dtranspose370 + x_offset,180,3,11,3,3,0,0,2,2,1,1,op_4_xcol);

       BLAS::sgemm_(&op_4_transA, &op_4_transB, &op_4_m, &op_4_n, &op_4_k, &op_4_alpha, op_4_xcol, &op_4_m,
         op_4_f, &op_4_k, &op_4_beta, tensor_ConvTranspose790 + out_offset, &op_4_m);
   }
   int op_4_size = 28980;
   float op_4_gamma = 1.0;
   int op_4_incx = 1;
   int op_4_incy = 1;
   BLAS::saxpy_(&op_4_size, &op_4_gamma, tensor_BroadcastedStatefulPartitionedCalldecoderconv2dtransposeBiasAddReadVariableOp0, &op_4_incx, tensor_ConvTranspose790, &op_4_incy);

//------ RELU
   for (int id = 0; id < 28980 ; id++){
      tensor_StatefulPartitionedCalldecoderconv2dtransposeRelu0[id] = ((tensor_ConvTranspose790[id] > 0 )? tensor_ConvTranspose790[id] : 0);
   }

//----  operator ConvTranspose op_6
   float * op_6_f = fVec_op_6_f.data();
   for (std::size_t oc = 0; oc < 180; oc++) {
      for (std::size_t ic = 0; ic < 90; ic++) {
         for (std::size_t kh = 0; kh < 3; kh++) {
            for (std::size_t kw = 0; kw < 3; kw++) {
               op_6_f[oc * 810 + ic * 9 + kh * 3 + kw * 1  ] = tensor_StatefulPartitionedCalldecoderconv2dtranspose1conv2dtransposeReadVariableOp0[oc * 810 + ic * 9 + kh * 3 + kw ];
            }
         }
      }
   }
   char op_6_transA = 'N';
   char op_6_transB = 'N';
   int op_6_m = 705;
   int op_6_n = 90;
   int op_6_k = 9;
   float op_6_alpha = 1.0;
   float op_6_beta = 0.0;
   float * op_6_xcol = fVec_op_6_xcol.data();
   for (size_t n = 0; n < 1; n++) {
      size_t x_offset = n * 28980;
      size_t out_offset = n * 63450;
      TMVA::Experimental::SOFIE::UTILITY::Im2col<float>(tensor_StatefulPartitionedCalldecoderconv2dtransposeRelu0 + x_offset,180,7,23,3,3,0,0,2,2,1,1,op_6_xcol);

       BLAS::sgemm_(&op_6_transA, &op_6_transB, &op_6_m, &op_6_n, &op_6_k, &op_6_alpha, op_6_xcol, &op_6_m,
         op_6_f, &op_6_k, &op_6_beta, tensor_StatefulPartitionedCalldecoderconv2dtranspose1conv2dtranspose0 + out_offset, &op_6_m);
   }

//------ Add
   for (size_t id = 0; id < 63450 ; id++){
      tensor_StatefulPartitionedCalldecoderconv2dtranspose1BiasAdd0[id] = tensor_constfoldopt121[id] + tensor_StatefulPartitionedCalldecoderconv2dtranspose1conv2dtranspose0[id] ;
   }

//------ RELU
   for (int id = 0; id < 63450 ; id++){
      tensor_StatefulPartitionedCalldecoderconv2dtranspose1Relu0[id] = ((tensor_StatefulPartitionedCalldecoderconv2dtranspose1BiasAdd0[id] > 0 )? tensor_StatefulPartitionedCalldecoderconv2dtranspose1BiasAdd0[id] : 0);
   }

//----  operator ConvTranspose op_9
   float * op_9_f = fVec_op_9_f.data();
   for (std::size_t oc = 0; oc < 90; oc++) {
      for (std::size_t ic = 0; ic < 45; ic++) {
         for (std::size_t kh = 0; kh < 4; kh++) {
            for (std::size_t kw = 0; kw < 4; kw++) {
               op_9_f[oc * 720 + ic * 16 + kh * 4 + kw * 1  ] = tensor_StatefulPartitionedCalldecoderconv2dtranspose2conv2dtransposeReadVariableOp0[oc * 720 + ic * 16 + kh * 4 + kw ];
            }
         }
      }
   }
   char op_9_transA = 'N';
   char op_9_transB = 'N';
   int op_9_m = 900;
   int op_9_n = 45;
   int op_9_k = 16;
   float op_9_alpha = 1.0;
   float op_9_beta = 0.0;
   float * op_9_xcol = fVec_op_9_xcol.data();
   for (size_t n = 0; n < 1; n++) {
      size_t x_offset = n * 63450;
      size_t out_offset = n * 40500;
      TMVA::Experimental::SOFIE::UTILITY::Im2col<float>(tensor_StatefulPartitionedCalldecoderconv2dtranspose1Relu0 + x_offset,90,15,47,4,4,0,0,1,1,1,1,op_9_xcol);

       BLAS::sgemm_(&op_9_transA, &op_9_transB, &op_9_m, &op_9_n, &op_9_k, &op_9_alpha, op_9_xcol, &op_9_m,
         op_9_f, &op_9_k, &op_9_beta, tensor_StatefulPartitionedCalldecoderconv2dtranspose2conv2dtranspose0 + out_offset, &op_9_m);
   }

//------ Add
   for (size_t id = 0; id < 40500 ; id++){
      tensor_StatefulPartitionedCalldecoderconv2dtranspose2BiasAdd0[id] = tensor_constfoldopt118[id] + tensor_StatefulPartitionedCalldecoderconv2dtranspose2conv2dtranspose0[id] ;
   }
	for (int id = 0; id < 40500 ; id++){
		tensor_StatefulPartitionedCalldecoderconv2dtranspose2Sigmoid0[id] = 1 / (1 + std::exp( - tensor_StatefulPartitionedCalldecoderconv2dtranspose2BiasAdd0[id]));
	}
   ///------- Transpose operator

   for (size_t id = 0; id < 40500 ; id++){
      tensor_conv2dtranspose2[id] = tensor_StatefulPartitionedCalldecoderconv2dtranspose2Sigmoid0[ ( id / 40500 ) * 40500 + ( (id % 45) ) * 900 + ( (id % 40500) / 2250 ) * 50 + ( (id % 2250) / 45 )];
   }
   std::vector<float> ret (tensor_conv2dtranspose2, tensor_conv2dtranspose2 + 40500);
   return ret;
}
};
} //TMVA_SOFIE_Conv2DTranspose_Relu_Sigmoid

#endif  // TMVA_SOFIE_CONV2DTRANSPOSE_RELU_SIGMOID
