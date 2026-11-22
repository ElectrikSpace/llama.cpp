#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// The MPPA device needs the runtime library built by xtc (mppa_runtime_host.so), its path is read from
// the XTC_MPPA_RUNTIME environment variable.
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xtc_mppa_reg(void);

#ifdef  __cplusplus
}
#endif
