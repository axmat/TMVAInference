execute_process(
COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/GenConvTranspose.py)
#execute_process(COMMAND root SofieConvTranspose.C)

add_executable(ConvTranspose ConvTranspose.cxx)
target_include_directories(ConvTranspose PRIVATE ${BLAS_INCLUDE})
target_link_libraries(ConvTranspose PUBLIC ROOT::ROOTTMVASofie ${BLAS_LIB})
