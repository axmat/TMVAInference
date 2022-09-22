#execute_process(
#COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/GenFuseConv.py)
#execute_process(COMMAND root SofieFuseConv.C)

add_executable(FuseConv FuseConv.cxx)
#target_include_directories(FuseConv PRIVATE ${BLAS_INCLUDE})
target_link_libraries(FuseConv PUBLIC ROOT::ROOTTMVASofie ${BLAS_LIBRARIES})
