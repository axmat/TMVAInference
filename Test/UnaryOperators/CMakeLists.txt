execute_process(
COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/GenUnaryOperators.py)
#execute_process(COMMAND root SofieUnaryOperators.C)

add_executable(UnaryOperators UnaryOperators.cxx)
#target_include_directories(UnaryOperators PRIVATE ${BLAS_INCLUDE})
target_link_libraries(UnaryOperators PUBLIC ROOT::ROOTTMVASofie)
