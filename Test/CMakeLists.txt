cmake_minimum_required(VERSION 3.2)

project(test)

find_package(ROOT REQUIRED COMPONENTS ROOTTMVASofie)
#find_package(Python REQUIRED)

# TODO find openblas
find_package(BLAS REQUIRED)
#set(BLAS_INCLUDE /usr/local/opt/openblas/include)
#set(BLAS_LIB /usr/local/opt/openblas/lib/libopenblas.dylib)

#add_subdirectory(ConvTranspose)
add_subdirectory(UnaryOperators)
add_subdirectory(FuseConvAdd)
