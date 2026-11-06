#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_xtc_init(void);

GGML_BACKEND_API bool ggml_backend_is_xtc(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_xtc_set_n_threads(ggml_backend_t backend_xtc, int n_threads);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xtc_reg(void);

#ifdef  __cplusplus
}
#endif
