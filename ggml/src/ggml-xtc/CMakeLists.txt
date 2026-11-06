if (GGML_BACKEND_DL)
    message(FATAL_ERROR "GGML_XTC requires GGML_BACKEND_DL=OFF (ggml-xtc links against ggml-cpu)")
endif()

if (NOT GGML_CPU)
    message(FATAL_ERROR "GGML_XTC requires GGML_CPU=ON")
endif()

ggml_add_backend_library(ggml-xtc
                         ggml-xtc.cpp)

target_link_libraries(ggml-xtc PRIVATE ggml-cpu)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(ggml-xtc PRIVATE dl)
endif()
