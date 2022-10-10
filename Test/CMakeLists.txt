cmake_minimum_required(VERSION 3.2)

project(test)

find_package(ROOT REQUIRED COMPONENTS ROOTTMVASofie)
#find_package(Python REQUIRED)

find_package(BLAS REQUIRED)

add_subdirectory(ConvTranspose)
add_subdirectory(UnaryOperators)
#add_subdirectory(FuseConvAdd)
#add_subdirectory(BroadcastBinaryOp)
