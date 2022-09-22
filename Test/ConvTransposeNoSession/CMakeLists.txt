#execute_process(
#COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/GenConvTranspose.py)
#execute_process(COMMAND root SofieConvTranspose.C)
cmake_minimum_required(VERSION 3.2)

project(test)

find_package(ROOT REQUIRED COMPONENTS ROOTTMVASofie)
#find_package(Python REQUIRED)

# TODO find openblas
find_package(BLAS REQUIRED)
#set(BLAS_INCLUDE /usr/include/x86_64-linux-gnu/cblas-openblas.h)
#set(BLAS_LIB /usr/local/opt/openblas/lib/libopenblas)

add_executable(ConvTranspose ConvTranspose.cxx)
#target_include_directories(ConvTranspose PRIVATE ${BLAS_INCLUDE})
target_link_libraries(ConvTranspose PUBLIC ROOT::ROOTTMVASofie ${BLAS_LIBRARIES})
