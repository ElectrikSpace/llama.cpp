if (GGML_BACKEND_DL)
    message(FATAL_ERROR "GGML_XTC requires GGML_BACKEND_DL=OFF (ggml-xtc links against ggml-cpu)")
endif()

if (NOT GGML_CPU)
    message(FATAL_ERROR "GGML_XTC requires GGML_CPU=ON")
endif()

add_library(ggml-xtc-common OBJECT ggml-xtc-common.cpp ggml-xtc-common.h)
target_link_libraries(ggml-xtc-common PRIVATE ggml-base xtc_support_lib)
target_include_directories(ggml-xtc-common PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(ggml-xtc-common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(ggml-xtc-common PROPERTIES POSITION_INDEPENDENT_CODE ON)

ggml_add_backend_library(ggml-xtc-cpu
                         ggml-xtc.cpp)
target_link_libraries(ggml-xtc-cpu PRIVATE ggml-xtc-common)

# xtc_support_lib lives next to llama.cpp (<project>/xtc_support_lib); override with XTC_SUPPORT_LIB_DIR
get_filename_component(XTC_SUPPORT_LIB_DEFAULT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../xtc_support_lib" ABSOLUTE)
set(XTC_SUPPORT_LIB_DIR "${XTC_SUPPORT_LIB_DEFAULT_DIR}" CACHE PATH "directory of xtc_support_lib")
add_subdirectory(${XTC_SUPPORT_LIB_DIR} ${CMAKE_CURRENT_BINARY_DIR}/xtc_support_lib)

target_link_libraries(ggml-xtc-cpu PRIVATE ggml-cpu xtc_support_lib)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(ggml-xtc-cpu PRIVATE dl)
endif()
