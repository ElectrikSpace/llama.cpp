# host independent part of the XTC backends (ggml-xtc-cpu, ggml-xtc-mppa), see ggml/src/ggml-xtc/ggml-xtc-common.h

# xtc_support_lib lives next to llama.cpp (<project>/xtc_support_lib); override with XTC_SUPPORT_LIB_DIR
get_filename_component(XTC_SUPPORT_LIB_DEFAULT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../xtc_support_lib" ABSOLUTE)
set(XTC_SUPPORT_LIB_DIR "${XTC_SUPPORT_LIB_DEFAULT_DIR}" CACHE PATH "directory of xtc_support_lib")
add_subdirectory(${XTC_SUPPORT_LIB_DIR} ${CMAKE_CURRENT_BINARY_DIR}/xtc_support_lib)

add_library(ggml-xtc-common OBJECT ../ggml-xtc/ggml-xtc-common.cpp ../ggml-xtc/ggml-xtc-common.h)
target_link_libraries(ggml-xtc-common PRIVATE ggml-base)
target_include_directories(ggml-xtc-common PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(ggml-xtc-common PUBLIC xtc_support_lib)
target_include_directories(ggml-xtc-common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../ggml-xtc)
set_target_properties(ggml-xtc-common PROPERTIES POSITION_INDEPENDENT_CODE ON)
