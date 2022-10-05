execute_process(
COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/GenBroadcastBinaryOp.py)

add_executable(BroadcastBinaryOp BroadcastBinaryOp.cxx)
#target_include_directories(BroadcastBinaryOp PRIVATE ${BLAS_INCLUDE})
target_link_libraries(BroadcastBinaryOp PUBLIC ROOT::ROOTTMVASofie)
