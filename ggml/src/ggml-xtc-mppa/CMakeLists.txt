if (GGML_BACKEND_DL)
    message(FATAL_ERROR "GGML_XTC_MPPA requires GGML_BACKEND_DL=OFF")
endif()

ggml_add_backend_library(ggml-xtc-mppa
                         ggml-xtc-mppa.cpp)
target_link_libraries(ggml-xtc-mppa PRIVATE ggml-xtc-common)
target_link_libraries(ggml-xtc-mppa PRIVATE ggml-xtc-common)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(ggml-xtc-mppa PRIVATE dl)
endif()
