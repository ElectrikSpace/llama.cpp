#include "ggml-xtc.h"

#include "ggml-xtc-common.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "xtc_kernel_abi.h"
#include "xtc_support_lib.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace fs = std::filesystem;

//
// XTC backend
//
// Every supported kernel (F32 MUL_MAT, and same-shape F32 ADD / MUL) is looked up in the XTC kernel base
// (see xtc/src/xtc/kernelbase). The key is the first 16 hex digits of the sha256 of the
// canonical XTC subgraph, which this file rebuilds as a JSON string. It must stay
// byte-identical to xtc.kernelbase.canonical.canonical_graph_json() for ggml_matmul_graph() and for an
// xtc add / mul of two (M, N) tensors, with M = ne1*ne2*ne3 and N = ne0.
//
// The other operators of a transformer layer (RMS_NORM, SWIGLU, ROPE, SET_ROWS, FLASH_ATTN_EXT, and
// ADD / MUL with broadcast) run through xtc_support_lib (<project>/xtc_support_lib/), plain C reference kernels. The
// backend has no embedded CPU backend: any other op is not claimed by the device.
//
// If a kernel is missing, graph_compute fails *before computing anything*, and a stub
// (graph.yaml + a build_kernel.py where only the schedule is missing) is written to <kernelbase>/_pending/<key>/.
// The user fills the schedule in build_kernel.py, runs it, then reruns llama.cpp.
//
// Kernels come from the bundle libxtc_kernels.so (XTC_KERNEL_LIB, default <kernelbase>/libxtc_kernels.so),
// whose wrappers check the tensors. Without a bundle entry, <kernelbase>/<key>/kernel.so is used (unchecked).
//
// Environment:
//   XTC_KERNELBASE           kernel base root (default: ./xtc_kernelbase)
//   XTC_KERNEL_LIB           kernel bundle (default: <kernelbase>/libxtc_kernels.so)
//   GGML_XTC_BUNDLE_ONLY     1 -> only the bundle is used, no per-entry kernel.so
//   GGML_XTC_MISSING_KERNEL  fail (default) | cpu   (cpu: the device does not claim matmuls without kernel, so
//                            the scheduler runs them on another backend, normally the CPU one)
//   GGML_XTC_DISABLE         1 -> the XTC device claims no op at all
//

struct ggml_backend_xtc_context {
    int n_threads = GGML_DEFAULT_N_THREADS; // unused: the support lib kernels are single-threaded
};

static bool ggml_xtc_is_disabled() {
    const char * env = getenv("GGML_XTC_DISABLE");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0;
}

static bool ggml_xtc_missing_falls_back_to_cpu() {
    const char * env = getenv("GGML_XTC_MISSING_KERNEL");
    return env != nullptr && strcmp(env, "cpu") == 0;
}

static bool ggml_xtc_lib_supports_op(const struct ggml_tensor * op);

static bool ggml_xtc_is_flat_eltwise(const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    return src0 != nullptr && src1 != nullptr && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(op) &&
           ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, op);
}

// all the rows of the node and of its sources are packed, so one kernel call can take them all
static bool ggml_xtc_flat_rows(const struct ggml_tensor * op) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = op->src[i];
        if (src != nullptr && src->type == GGML_TYPE_F32 && src->ne[0] > 1 && !ggml_is_contiguous(src)) {
            return false;
        }
    }
    return ggml_is_contiguous(op);
}

// Kernels needed by a node, in call order. Empty when xtc_support_lib runs the node.
static std::vector<ggml_xtc_kernel_shape> ggml_xtc_node_kernels(const struct ggml_tensor * op) {
    std::vector<ggml_xtc_kernel_shape> out;
    ggml_xtc_kernel_shape s;
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    switch (op->op) {
        case GGML_OP_MUL_MAT:
            if (ggml_xtc_is_supported_matmul(op, &s)) {
                out.push_back(s);
            }
            break;
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            s.kop = op->op == GGML_OP_ADD ? XTC_KOP_ADD : XTC_KOP_MUL;
            s.n   = op->ne[0];
            if (ggml_nelements(op) > 0 && ggml_xtc_is_flat_eltwise(op)) {
                s.m = op->ne[1] * op->ne[2] * op->ne[3];
                out.push_back(s);
            } else if (ggml_xtc_lib_supports_op(op) && src1->ne[0] == op->ne[0]) {
                s.m = 1; // one call per row of dst
                out.push_back(s);
            }
            break;
        case GGML_OP_ROPE:
            // x * cos + swap_pairs(x) * sin, one token (n_heads rows) at a time
            if (ggml_xtc_lib_supports_op(op) && src0->nb[1] == src0->ne[0] * sizeof(float) &&
                op->nb[1] == op->ne[0] * sizeof(float)) {
                const int64_t H = op->ne[1], D = op->ne[0];
                out.push_back({ XTC_KOP_MATMUL, H, D, D });
                out.push_back({ XTC_KOP_MUL, H, D, 0 });
                out.push_back({ XTC_KOP_FMA, H, D, 0 });
            }
            break;
        case GGML_OP_RMS_NORM:
            if (ggml_xtc_lib_supports_op(op)) {
                // x * rsqrt(mean(x^2) + eps); the row sum and the broadcast are matmuls with a vector of constants
                const int64_t M = ggml_xtc_flat_rows(op) ? ggml_nrows(op) : 1;
                const int64_t N = op->ne[0];
                out.push_back({ XTC_KOP_MUL, M, N, 0 });
                out.push_back({ XTC_KOP_MATMUL, M, 1, N });
                out.push_back({ XTC_KOP_ADD, M, 1, 0 });
                out.push_back({ XTC_KOP_RSQRT, M, 1, 0 });
                out.push_back({ XTC_KOP_MATMUL, M, N, 1 });
                out.push_back({ XTC_KOP_MUL, M, N, 0 });
            }
            break;
        case GGML_OP_GLU:
            if (ggml_xtc_lib_supports_op(op)) {
                const int64_t M = ggml_xtc_flat_rows(op) ? ggml_nrows(op) : 1;
                out.push_back({ XTC_KOP_SILU, M, op->ne[0], 0 });
                out.push_back({ XTC_KOP_MUL, M, op->ne[0], 0 });
            }
            break;
        case GGML_OP_SET_ROWS:
            if (op->type == GGML_TYPE_F16 && ggml_xtc_lib_supports_op(op)) {
                out.push_back({ XTC_KOP_CAST_F16, 1, op->ne[0], 0 });
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            // standard attention over the whole KV length N (see ggml_xtc_run_flash_attn_kernels)
            if (ggml_xtc_lib_supports_op(op) && src1->ne[1] > 0) {
                const int64_t M = (src0->ne[2] / src1->ne[2]) * src0->ne[1];
                const int64_t D = src0->ne[0];
                const int64_t N = src1->ne[1];
                out.push_back({ XTC_KOP_CAST_F16, M, D, 0 });
                out.push_back({ XTC_KOP_MATMUL_F16_BT, M, N, D });
                out.push_back({ XTC_KOP_FMA, M, N, 0 });
                out.push_back({ XTC_KOP_REDUCE_MAX, M, N, 0 });
                out.push_back({ XTC_KOP_SUB_ROW, M, N, 0 });
                out.push_back({ XTC_KOP_EXP, M, N, 0 });
                out.push_back({ XTC_KOP_REDUCE_SUM, M, N, 0 });
                out.push_back({ XTC_KOP_DIV_ROW, M, N, 0 });
                out.push_back({ XTC_KOP_CAST_F16, M, N, 0 });
                out.push_back({ XTC_KOP_MATMUL_F16, M, D, N });
            }
            break;
        default:
            break;
    }
    return out;
}

//
// graph helpers
//

static bool ggml_xtc_node_is_noop(const struct ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

// packed 2D view (ne0 contiguous, ne1 rows) of data, as the kernel wrappers expect
static ggml_tensor ggml_xtc_view2d(enum ggml_type type, int64_t ne0, int64_t ne1, const void * data) {
    ggml_tensor t = {};
    t.type  = type;
    t.data  = const_cast<void *>(data);
    t.ne[0] = ne0;
    t.ne[1] = ne1;
    t.ne[2] = 1;
    t.ne[3] = 1;
    t.nb[0] = ggml_type_size(type);
    t.nb[1] = t.nb[0] * ne0;
    t.nb[2] = t.nb[1] * ne1;
    t.nb[3] = t.nb[2];
    return t;
}

// srcs in kernel input order, dst the output. A mismatch between the tensors and the kernel is a backend bug.
static void ggml_xtc_invoke(const ggml_xtc_kernel & k, std::initializer_list<const ggml_tensor *> srcs,
                            ggml_tensor * dst) {
    GGML_ASSERT((int) srcs.size() == k.n_inputs);
    if (k.call != nullptr) {
        if (k.call(srcs.begin(), dst) != 0) {
            GGML_ABORT("%s: tensors do not match XTC kernel %s", __func__, k.key.c_str());
        }
        return;
    }
    void * p[4];
    int n = 0;
    for (const ggml_tensor * s : srcs) {
        p[n++] = s->data;
    }
    p[n] = dst->data;
    switch (n) {
        case 1: ((void (*)(void *, void *)) k.fn)(p[0], p[1]); break;
        case 2: k.fn(p[0], p[1], p[2]); break;
        case 3: ((void (*)(void *, void *, void *, void *)) k.fn)(p[0], p[1], p[2], p[3]); break;
        default: GGML_ABORT("%s: unexpected arity %d", __func__, n);
    }
}

static void ggml_xtc_run_mul_mat_kernel(const ggml_xtc_kernel & k, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    // A compiled kernel computes one 2D matmul (static M, N, K). ggml's MUL_MAT is batched over
    // ne[2] and ne[3]: src1 (activations, [K, M, ne12, ne13]) and dst ([N, M, ne12, ne13]) have
    // ne12 * ne13 slices, so the kernel is called once per slice.
    // src0 (weights, [K, N, ne02, ne03]) may have fewer slices: ggml broadcasts it, and the same
    // weight slice is reused for r2 = ne12/ne02 (resp. r3 = ne13/ne03) consecutive src1 slices.
    // Without batching (ne2 = ne3 = 1, e.g. every weight matmul of a layer) there is a single call.
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            // slice of src0 used by this slice of src1
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;

            const void * a = (const char *) src1->data + i12 * nb12 + i13 * nb13;
            const void * b = (const char *) src0->data + i02 * nb02 + i03 * nb03;
            void *       c = (char *) dst->data + i12 * nb2 + i13 * nb3;

            // B is stored transposed: (K, N) with ne0 = K
            ggml_tensor va = ggml_xtc_view2d(GGML_TYPE_F32, ne10, ne11, a);
            ggml_tensor vb = ggml_xtc_view2d(GGML_TYPE_F32, ne00, ne01, b);
            ggml_tensor vc = ggml_xtc_view2d(GGML_TYPE_F32, ne0, ne1, c);
            ggml_xtc_invoke(k, { &va, &vb }, &vc);
        }
    }
}

//
// operators implemented by xtc_support_lib (plain C reference kernels)
//

static inline bool ggml_xtc_f32_rows(const struct ggml_tensor * t) {
    return t != nullptr && t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float);
}

static bool ggml_xtc_lib_supports_op(const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];

    if (ggml_nelements(op) == 0) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_RMS_NORM:
            return ggml_xtc_f32_rows(src0) && ggml_xtc_f32_rows(op) && ggml_are_same_shape(src0, op);

        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return ggml_xtc_f32_rows(src0) && ggml_xtc_f32_rows(src1) && ggml_xtc_f32_rows(op) &&
                   ggml_are_same_shape(src0, op) && ggml_can_repeat(src1, src0);

        case GGML_OP_GLU:
            // split SWIGLU only (gate = src0, up = src1)
            return ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && ggml_get_op_params_i32(op, 1) == 0 &&
                   ggml_xtc_f32_rows(src0) && ggml_xtc_f32_rows(src1) && ggml_xtc_f32_rows(op) &&
                   ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, op);

        case GGML_OP_ROPE: {
            const int   n_dims = ggml_get_op_params_i32(op, 1);
            const int   mode   = ggml_get_op_params_i32(op, 2);
            float ext_factor;
            memcpy(&ext_factor, (const int32_t *) op->op_params + 7, sizeof(float));
            return ggml_xtc_f32_rows(src0) && ggml_xtc_f32_rows(op) && ggml_are_same_shape(src0, op) &&
                   src1 != nullptr && src1->type == GGML_TYPE_I32 && src1->ne[0] == src0->ne[2] &&
                   (mode == GGML_ROPE_TYPE_NORMAL || mode == GGML_ROPE_TYPE_NEOX) &&
                   ext_factor == 0.0f && ggml_get_op_params_i32(op, 15) == 0 &&
                   n_dims > 0 && n_dims % 2 == 0 && n_dims <= src0->ne[0] && n_dims <= XTC_MAX_HEAD_DIM &&
                   (src2 == nullptr || (src2->type == GGML_TYPE_F32 && src2->ne[0] >= n_dims / 2));
        }

        case GGML_OP_SET_ROWS:
            return ggml_xtc_f32_rows(src0) && src1 != nullptr && src1->type == GGML_TYPE_I64 &&
                   (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32) &&
                   op->nb[0] == ggml_type_size(op->type) && op->ne[0] == src0->ne[0] &&
                   op->ne[2] == src0->ne[2] && op->ne[3] == src0->ne[3] &&
                   src0->ne[2] % src1->ne[1] == 0 && src0->ne[3] % src1->ne[2] == 0 &&
                   ggml_is_contiguous_rows(src1);

        case GGML_OP_FLASH_ATTN_EXT: {
            const struct ggml_tensor * k    = src1;
            const struct ggml_tensor * v    = src2;
            const struct ggml_tensor * mask = op->src[3];
            float max_bias, softcap;
            memcpy(&max_bias, (const int32_t *) op->op_params + 1, sizeof(float));
            memcpy(&softcap,  (const int32_t *) op->op_params + 2, sizeof(float));
            for (int i = 4; i < GGML_MAX_SRC; i++) {
                if (op->src[i] != nullptr) { // sinks, kv_rows
                    return false;
                }
            }
            return ggml_xtc_f32_rows(src0) && ggml_xtc_f32_rows(op) &&
                   k != nullptr && v != nullptr && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 &&
                   k->nb[0] == sizeof(uint16_t) && v->nb[0] == sizeof(uint16_t) &&
                   max_bias == 0.0f && softcap == 0.0f &&
                   src0->ne[0] == k->ne[0] && k->ne[0] == v->ne[0] && src0->ne[0] <= XTC_MAX_HEAD_DIM &&
                   k->ne[1] == v->ne[1] && k->ne[2] == v->ne[2] && k->ne[3] == src0->ne[3] && v->ne[3] == src0->ne[3] &&
                   k->ne[2] > 0 && src0->ne[2] % k->ne[2] == 0 &&
                   (mask == nullptr || (mask->type == GGML_TYPE_F16 && mask->nb[0] == sizeof(uint16_t) &&
                                        mask->ne[0] >= k->ne[1] && mask->ne[1] >= src0->ne[1] &&
                                        src0->ne[2] % mask->ne[2] == 0 && src0->ne[3] % mask->ne[3] == 0));
        }

        default:
            return false;
    }
}

static xtc_view ggml_xtc_view(const struct ggml_tensor * t) {
    xtc_view v;
    v.data = t->data;
    for (int i = 0; i < 4; i++) {
        v.ne[i] = t->ne[i];
        v.nb[i] = t->nb[i];
    }
    return v;
}

static void ggml_xtc_run_lib_op(struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * src2 = dst->src[2];

    xtc_view d  = ggml_xtc_view(dst);
    xtc_view s0 = ggml_xtc_view(src0);

    switch (dst->op) {
        case GGML_OP_RMS_NORM: {
            float eps;
            memcpy(&eps, dst->op_params, sizeof(float));
            xtc_rms_norm_f32(&s0, &d, eps);
        } break;
        case GGML_OP_ADD: {
            xtc_view s1 = ggml_xtc_view(src1);
            xtc_add_f32(&s0, &s1, &d);
        } break;
        case GGML_OP_MUL: {
            xtc_view s1 = ggml_xtc_view(src1);
            xtc_mul_f32(&s0, &s1, &d);
        } break;
        case GGML_OP_GLU: {
            xtc_view s1 = ggml_xtc_view(src1);
            xtc_swiglu_f32(&s0, &s1, &d);
        } break;
        case GGML_OP_ROPE: {
            xtc_rope_params p;
            p.n_dims = ggml_get_op_params_i32(dst, 1);
            p.neox   = ggml_get_op_params_i32(dst, 2) == GGML_ROPE_TYPE_NEOX;
            memcpy(&p.freq_base,   (const int32_t *) dst->op_params + 5, sizeof(float));
            memcpy(&p.freq_scale,  (const int32_t *) dst->op_params + 6, sizeof(float));
            memcpy(&p.attn_factor, (const int32_t *) dst->op_params + 8, sizeof(float));
            xtc_rope_f32(&s0, (const int32_t *) src1->data, src2 ? (const float *) src2->data : nullptr, &d, &p);
        } break;
        case GGML_OP_SET_ROWS: {
            xtc_view idx = ggml_xtc_view(src1);
            xtc_set_rows_f32(&s0, &idx, &d, dst->type == GGML_TYPE_F16);
        } break;
        case GGML_OP_FLASH_ATTN_EXT: {
            xtc_view k = ggml_xtc_view(src1);
            xtc_view v = ggml_xtc_view(src2);
            xtc_view m;
            const struct ggml_tensor * mask = dst->src[3];
            if (mask) {
                m = ggml_xtc_view(mask);
            }
            float scale;
            memcpy(&scale, dst->op_params, sizeof(float));
            xtc_flash_attn_f16kv(&s0, &k, &v, mask ? &m : nullptr, &d, scale);
        } break;
        default:
            GGML_ABORT("%s: unexpected op %s", __func__, ggml_op_name(dst->op));
    }
}

// when the node is not flat, kernels are called once per row (M = 1)
template <typename F>
static void ggml_xtc_for_each_chunk(const struct ggml_tensor * dst, F && f) {
    if (ggml_xtc_flat_rows(dst)) {
        f(0, 0, 0);
        return;
    }
    for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
                f(i1, i2, i3);
            }
        }
    }
}

static void * ggml_xtc_row_ptr(const struct ggml_tensor * t, int64_t i1, int64_t i2, int64_t i3) {
    return (char *) t->data + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
}

// kernels: x*x, row sums (matmul by 1/N), + eps, rsqrt, broadcast (matmul by ones), x * scale
static void ggml_xtc_run_rms_norm_kernels(const std::vector<const ggml_xtc_kernel *> & fns, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t N = dst->ne[0];
    const int64_t M = ggml_xtc_flat_rows(dst) ? ggml_nrows(dst) : 1;
    std::vector<float> inv_n(N, 1.0f / (float) N), ones(N, 1.0f), eps_v(M, eps);
    std::vector<float> sq(M * N), bc(M * N), mean(M), mean_eps(M), rs(M);

    ggml_xtc_for_each_chunk(dst, [&](int64_t i1, int64_t i2, int64_t i3) {
        const void * x  = ggml_xtc_row_ptr(src0, i1, i2, i3);
        void *       out = ggml_xtc_row_ptr(dst, i1, i2, i3);
        ggml_tensor vx  = ggml_xtc_view2d(GGML_TYPE_F32, N, M, x);
        ggml_tensor vsq = ggml_xtc_view2d(GGML_TYPE_F32, N, M, sq.data());
        ggml_tensor vinv = ggml_xtc_view2d(GGML_TYPE_F32, N, 1, inv_n.data());
        ggml_tensor vmean = ggml_xtc_view2d(GGML_TYPE_F32, 1, M, mean.data());
        ggml_tensor veps = ggml_xtc_view2d(GGML_TYPE_F32, 1, M, eps_v.data());
        ggml_tensor vmean_eps = ggml_xtc_view2d(GGML_TYPE_F32, 1, M, mean_eps.data());
        ggml_tensor vrs = ggml_xtc_view2d(GGML_TYPE_F32, 1, M, rs.data());
        ggml_tensor vones = ggml_xtc_view2d(GGML_TYPE_F32, 1, N, ones.data());
        ggml_tensor vbc = ggml_xtc_view2d(GGML_TYPE_F32, N, M, bc.data());
        ggml_tensor vout = ggml_xtc_view2d(GGML_TYPE_F32, N, M, out);

        ggml_xtc_invoke(*fns[0], { &vx, &vx }, &vsq);
        ggml_xtc_invoke(*fns[1], { &vsq, &vinv }, &vmean);
        ggml_xtc_invoke(*fns[2], { &vmean, &veps }, &vmean_eps);
        ggml_xtc_invoke(*fns[3], { &vmean_eps }, &vrs);
        ggml_xtc_invoke(*fns[4], { &vrs, &vones }, &vbc);
        ggml_xtc_invoke(*fns[5], { &vx, &vbc }, &vout);
    });
}

// kernels: silu(gate), then * up
static void ggml_xtc_run_swiglu_kernels(const std::vector<const ggml_xtc_kernel *> & fns, struct ggml_tensor * dst) {
    const struct ggml_tensor * gate = dst->src[0];
    const struct ggml_tensor * up   = dst->src[1];
    const int64_t N = dst->ne[0];
    const int64_t M = ggml_xtc_flat_rows(dst) ? ggml_nrows(dst) : 1;
    std::vector<float> tmp(M * N);

    ggml_xtc_for_each_chunk(dst, [&](int64_t i1, int64_t i2, int64_t i3) {
        ggml_tensor vgate = ggml_xtc_view2d(GGML_TYPE_F32, N, M, ggml_xtc_row_ptr(gate, i1, i2, i3));
        ggml_tensor vtmp  = ggml_xtc_view2d(GGML_TYPE_F32, N, M, tmp.data());
        ggml_tensor vup   = ggml_xtc_view2d(GGML_TYPE_F32, N, M, ggml_xtc_row_ptr(up, i1, i2, i3));
        ggml_tensor vout  = ggml_xtc_view2d(GGML_TYPE_F32, N, M, ggml_xtc_row_ptr(dst, i1, i2, i3));
        ggml_xtc_invoke(*fns[0], { &vgate }, &vtmp);
        ggml_xtc_invoke(*fns[1], { &vtmp, &vup }, &vout);
    });
}

// kernel: F32 -> F16 cast of one row into the destination row
static void ggml_xtc_run_set_rows_kernels(const std::vector<const ggml_xtc_kernel *> & fns, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * idx  = dst->src[1];
    const int64_t N = src0->ne[0];
    for (int64_t i3 = 0; i3 < src0->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < src0->ne[2]; i2++) {
            for (int64_t i = 0; i < src0->ne[1]; i++) {
                const int64_t row = *(const int64_t *) ((const char *) idx->data + i * idx->nb[0] +
                                                        (i2 % idx->ne[1]) * idx->nb[1] + (i3 % idx->ne[2]) * idx->nb[2]);
                ggml_tensor vsrc = ggml_xtc_view2d(GGML_TYPE_F32, N, 1, ggml_xtc_row_ptr(src0, i, i2, i3));
                ggml_tensor vdst = ggml_xtc_view2d(GGML_TYPE_F16, N, 1, ggml_xtc_row_ptr(dst, row, i2, i3));
                ggml_xtc_invoke(*fns[0], { &vsrc }, &vdst);
            }
        }
    }
}

// one call of the (1, N) or (M, N) kernel on the whole tensor when it is flat, else one call per row of dst
static void ggml_xtc_run_eltwise_kernel(const ggml_xtc_kernel & k, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int64_t N = dst->ne[0];

    if (ggml_xtc_is_flat_eltwise(dst)) {
        ggml_tensor va = ggml_xtc_view2d(GGML_TYPE_F32, N, ggml_nrows(dst), src0->data);
        ggml_tensor vb = ggml_xtc_view2d(GGML_TYPE_F32, N, ggml_nrows(dst), src1->data);
        ggml_tensor vd = ggml_xtc_view2d(GGML_TYPE_F32, N, ggml_nrows(dst), dst->data);
        ggml_xtc_invoke(k, { &va, &vb }, &vd);
        return;
    }
    for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
                const void * a = ggml_xtc_row_ptr(src0, i1, i2, i3);
                const void * b = (char *) src1->data + (i1 % src1->ne[1]) * src1->nb[1] + (i2 % src1->ne[2]) * src1->nb[2] +
                                 (i3 % src1->ne[3]) * src1->nb[3];
                ggml_tensor va = ggml_xtc_view2d(GGML_TYPE_F32, N, 1, a);
                ggml_tensor vb = ggml_xtc_view2d(GGML_TYPE_F32, N, 1, b);
                ggml_tensor vd = ggml_xtc_view2d(GGML_TYPE_F32, N, 1, ggml_xtc_row_ptr(dst, i1, i2, i3));
                ggml_xtc_invoke(k, { &va, &vb }, &vd);
            }
        }
    }
}

// kernels: matmul by the swap matrix, mul by sin, fma with cos (see ggml_xtc_node_kernels)
static void ggml_xtc_run_rope_kernels(const std::vector<const ggml_xtc_kernel *> & fns, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src2 = dst->src[2];
    const int32_t * pos = (const int32_t *) dst->src[1]->data;

    xtc_rope_params p;
    p.n_dims = ggml_get_op_params_i32(dst, 1);
    p.neox   = ggml_get_op_params_i32(dst, 2) == GGML_ROPE_TYPE_NEOX;
    memcpy(&p.freq_base,   (const int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&p.freq_scale,  (const int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&p.attn_factor, (const int32_t *) dst->op_params + 8, sizeof(float));
    const float * ff = src2 ? (const float *) src2->data : nullptr;

    const int64_t D = dst->ne[0];
    const int64_t H = dst->ne[1];
    std::vector<float> swap(D * D), c(D), s(D), C(H * D), S(H * D), t1(H * D), t2(H * D);
    xtc_rope_swap_matrix(&p, D, swap.data());

    ggml_tensor vswap = ggml_xtc_view2d(GGML_TYPE_F32, D, D, swap.data());
    for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
            xtc_rope_factors(&p, D, pos[i2], ff, c.data(), s.data());
            for (int64_t h = 0; h < H; h++) {
                memcpy(&C[h * D], c.data(), D * sizeof(float));
                memcpy(&S[h * D], s.data(), D * sizeof(float));
            }
            const void * x   = (const char *) src0->data + i2 * src0->nb[2] + i3 * src0->nb[3];
            void *       out = (char *) dst->data + i2 * dst->nb[2] + i3 * dst->nb[3];
            ggml_tensor vx   = ggml_xtc_view2d(GGML_TYPE_F32, D, H, x);
            ggml_tensor vt1  = ggml_xtc_view2d(GGML_TYPE_F32, D, H, t1.data());
            ggml_tensor vS   = ggml_xtc_view2d(GGML_TYPE_F32, D, H, S.data());
            ggml_tensor vt2  = ggml_xtc_view2d(GGML_TYPE_F32, D, H, t2.data());
            ggml_tensor vC   = ggml_xtc_view2d(GGML_TYPE_F32, D, H, C.data());
            ggml_tensor vout = ggml_xtc_view2d(GGML_TYPE_F32, D, H, out);
            ggml_xtc_invoke(*fns[0], { &vx, &vswap }, &vt1);
            ggml_xtc_invoke(*fns[1], { &vt1, &vS }, &vt2);
            ggml_xtc_invoke(*fns[2], { &vx, &vC, &vt2 }, &vout);
        }
    }
}

// Standard attention, per (sequence, KV head): the queries of the GQA group are the M rows of the kernels.
//   q16 = F16(Q); S = q16.K^T; S = S * scale + mask; P = softmax_rows(S); out = F16(P).V
// Like the CPU backend, Q and P are rounded to F16. The kernels are listed in ggml_xtc_node_kernels.
static void ggml_xtc_run_flash_attn_kernels(const std::vector<const ggml_xtc_kernel *> & fns, struct ggml_tensor * dst) {
    const struct ggml_tensor * q    = dst->src[0];
    const struct ggml_tensor * k    = dst->src[1];
    const struct ggml_tensor * v    = dst->src[2];
    const struct ggml_tensor * mask = dst->src[3];
    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const int64_t D     = q->ne[0];
    const int64_t n_tok = q->ne[1];
    const int64_t n_kv  = k->ne[1];
    const int64_t rk2   = q->ne[2] / k->ne[2];
    const int64_t M     = rk2 * n_tok;

    std::vector<float>       qf(M * D), S(M * n_kv), scale_v(M * n_kv, scale), mask_f(M * n_kv, 0.0f), tmp(M * n_kv);
    std::vector<float>       row_max(M), row_sum(M), O(M * D);
    std::vector<ggml_fp16_t> q16(M * D), p16(M * n_kv), k_g, v_g;

    for (int64_t s = 0; s < q->ne[3]; s++) {
        for (int64_t g = 0; g < k->ne[2]; g++) {
            for (int64_t r = 0; r < M; r++) {
                const int64_t h = g * rk2 + r / n_tok;
                const int64_t t = r % n_tok;
                memcpy(&qf[r * D], (const char *) q->data + t * q->nb[1] + h * q->nb[2] + s * q->nb[3], D * sizeof(float));
                if (mask) {
                    const ggml_fp16_t * mp = (const ggml_fp16_t *) ((const char *) mask->data + t * mask->nb[1] +
                        (h % mask->ne[2]) * mask->nb[2] + (s % mask->ne[3]) * mask->nb[3]);
                    for (int64_t j = 0; j < n_kv; j++) {
                        mask_f[r * n_kv + j] = ggml_fp16_to_fp32(mp[j]);
                    }
                }
            }

            // K and V of this head, (n_kv, D) F16 rows: used in place when they are packed in the cache
            const char * kp = (const char *) k->data + g * k->nb[2] + s * k->nb[3];
            const char * vp = (const char *) v->data + g * v->nb[2] + s * v->nb[3];
            if (k->nb[1] != D * sizeof(ggml_fp16_t)) {
                k_g.resize(n_kv * D);
                for (int64_t j = 0; j < n_kv; j++) {
                    memcpy(&k_g[j * D], kp + j * k->nb[1], D * sizeof(ggml_fp16_t));
                }
                kp = (const char *) k_g.data();
            }
            if (v->nb[1] != D * sizeof(ggml_fp16_t)) {
                v_g.resize(n_kv * D);
                for (int64_t j = 0; j < n_kv; j++) {
                    memcpy(&v_g[j * D], vp + j * v->nb[1], D * sizeof(ggml_fp16_t));
                }
                vp = (const char *) v_g.data();
            }

            ggml_tensor vqf   = ggml_xtc_view2d(GGML_TYPE_F32, D, M, qf.data());
            ggml_tensor vq16  = ggml_xtc_view2d(GGML_TYPE_F16, D, M, q16.data());
            ggml_tensor vk    = ggml_xtc_view2d(GGML_TYPE_F16, D, n_kv, kp);
            ggml_tensor vv    = ggml_xtc_view2d(GGML_TYPE_F16, D, n_kv, vp);
            ggml_tensor vS    = ggml_xtc_view2d(GGML_TYPE_F32, n_kv, M, S.data());
            ggml_tensor vscale = ggml_xtc_view2d(GGML_TYPE_F32, n_kv, M, scale_v.data());
            ggml_tensor vmask = ggml_xtc_view2d(GGML_TYPE_F32, n_kv, M, mask_f.data());
            ggml_tensor vmax  = ggml_xtc_view2d(GGML_TYPE_F32, M, 1, row_max.data());
            ggml_tensor vtmp  = ggml_xtc_view2d(GGML_TYPE_F32, n_kv, M, tmp.data());
            ggml_tensor vsum  = ggml_xtc_view2d(GGML_TYPE_F32, M, 1, row_sum.data());
            ggml_tensor vp16  = ggml_xtc_view2d(GGML_TYPE_F16, n_kv, M, p16.data());
            ggml_tensor vO    = ggml_xtc_view2d(GGML_TYPE_F32, D, M, O.data());

            ggml_xtc_invoke(*fns[0], { &vqf }, &vq16);                                // q16 = F16(Q)
            ggml_xtc_invoke(*fns[1], { &vq16, &vk }, &vS);                            // S = q16.K^T
            ggml_xtc_invoke(*fns[2], { &vS, &vscale, &vmask }, &vS);                  // S = S * scale + mask
            ggml_xtc_invoke(*fns[3], { &vS }, &vmax);                                 // row max
            for (int64_t r = 0; r < M; r++) {
                if (row_max[r] == -INFINITY) { // fully masked row: exp(-inf) = 0 and a zero output
                    row_max[r] = 0.0f;
                }
            }
            ggml_xtc_invoke(*fns[4], { &vS, &vmax }, &vtmp);                          // S - max
            ggml_xtc_invoke(*fns[5], { &vtmp }, &vtmp);                               // exp
            ggml_xtc_invoke(*fns[6], { &vtmp }, &vsum);                               // row sum
            for (int64_t r = 0; r < M; r++) {
                if (row_sum[r] == 0.0f) {
                    row_sum[r] = 1.0f;
                }
            }
            ggml_xtc_invoke(*fns[7], { &vtmp, &vsum }, &vtmp);                        // P
            ggml_xtc_invoke(*fns[8], { &vtmp }, &vp16);                               // F16(P)
            ggml_xtc_invoke(*fns[9], { &vp16, &vv }, &vO);                            // out = P.V

            for (int64_t r = 0; r < M; r++) {
                const int64_t h = g * rk2 + r / n_tok;
                const int64_t t = r % n_tok;
                memcpy((char *) dst->data + h * dst->nb[1] + t * dst->nb[2] + s * dst->nb[3], &O[r * D], D * sizeof(float));
            }
        }
    }
}

enum ggml_xtc_op_kind {
    GGML_XTC_OP_UNSUPPORTED = 0,
    GGML_XTC_OP_KERNEL      = 1, // XTC-compiled kernel (kernel base)
    GGML_XTC_OP_LIB         = 2, // xtc_support_lib
    GGML_XTC_OP_VIEW        = 3, // metadata only
};

static enum ggml_xtc_op_kind ggml_xtc_classify_op(const struct ggml_tensor * op) {
    if (ggml_xtc_is_disabled()) {
        return GGML_XTC_OP_UNSUPPORTED;
    }
    if (ggml_xtc_node_is_noop(op)) {
        return GGML_XTC_OP_VIEW;
    }
    if (!ggml_xtc_node_kernels(op).empty()) {
        return GGML_XTC_OP_KERNEL;
    }
    if (ggml_xtc_lib_supports_op(op)) {
        return GGML_XTC_OP_LIB;
    }
    return GGML_XTC_OP_UNSUPPORTED;
}

// The XTC device claims only what XTC (matmul kernels) or xtc_support_lib run: everything else
// is left to the scheduler (another backend, normally the CPU one).
static bool ggml_xtc_device_supports_op(const struct ggml_tensor * op) {
    switch (ggml_xtc_classify_op(op)) {
        case GGML_XTC_OP_KERNEL:
            if (!ggml_xtc_missing_falls_back_to_cpu()) {
                // claim every supported node: a missing kernel is a clean failure in graph_compute
                return true;
            }
            {
                std::lock_guard<std::mutex> lock(g_xtc_mutex);
                for (const auto & shape : ggml_xtc_node_kernels(op)) {
                    if (ggml_xtc_get_kernel(shape).state != ggml_xtc_kernel_state::FOUND) {
                        return false;
                    }
                }
                return true;
            }
        case GGML_XTC_OP_LIB:
        case GGML_XTC_OP_VIEW:
            return true;
        default:
            return false;
    }
}

// backend interface

static const char * ggml_backend_xtc_get_name(ggml_backend_t backend) {
    return "XTC";

    GGML_UNUSED(backend);
}

static void ggml_backend_xtc_free(ggml_backend_t backend) {
    delete (ggml_backend_xtc_context *) backend->context;
    delete backend;
}

static enum ggml_status ggml_backend_xtc_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);

    // pass 1: resolve every kernel *before* computing anything, so that a missing kernel
    // fails the whole graph cleanly (and we can report all the missing ones at once)
    std::vector<std::vector<const ggml_xtc_kernel *>> fns(cgraph->n_nodes);
    std::vector<ggml_xtc_missing_entry>  missing;
    std::vector<std::string>             broken;
    {
        std::lock_guard<std::mutex> lock(g_xtc_mutex);
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];
            if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_xtc_node_is_noop(node)) {
                continue;
            }

            const std::vector<ggml_xtc_kernel_shape> shapes = ggml_xtc_node_kernels(node);
            if (shapes.empty()) {
                if (ggml_xtc_classify_op(node) != GGML_XTC_OP_LIB) {
                    GGML_LOG_ERROR("%s: unsupported op %s (%s)\n", __func__, ggml_op_desc(node), node->name);
                    return GGML_STATUS_FAILED;
                }
                continue;
            }

            for (const auto & shape : shapes) {
                const ggml_xtc_kernel & k = ggml_xtc_get_kernel(shape);
                switch (k.state) {
                    case ggml_xtc_kernel_state::FOUND:
                        fns[i].push_back(&k);
                        break;
                    case ggml_xtc_kernel_state::BROKEN:
                        broken.push_back(k.key + ": " + k.error);
                        break;
                    case ggml_xtc_kernel_state::MISSING: {
                        bool dup = false;
                        for (const auto & e : missing) {
                            dup = dup || e.key == k.key;
                        }
                        if (!dup) {
                            ggml_xtc_missing_entry e;
                            e.key       = k.key;
                            e.shape     = shape;
                            e.src0_name = node->src[0]->name;
                            missing.push_back(std::move(e));
                        }
                    } break;
                }
            }
        }
    }

    if (!missing.empty() || !broken.empty()) {
        const fs::path root = ggml_xtc_kernelbase_root();
        for (auto & e : missing) {
            ggml_xtc_write_stub(root, e);
        }
        GGML_LOG_ERROR("%s", ggml_xtc_failure_message(root, missing, broken).c_str());
        return GGML_STATUS_FAILED;
    }

    // pass 2: XTC kernels for the matmuls, xtc_support_lib for the other operators
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_xtc_node_is_noop(node)) {
            continue;
        }
        if (fns[i].empty()) {
            ggml_xtc_run_lib_op(node);
        } else if (node->op == GGML_OP_MUL_MAT) {
            ggml_xtc_run_mul_mat_kernel(*fns[i][0], node);
        } else if (node->op == GGML_OP_ROPE) {
            ggml_xtc_run_rope_kernels(fns[i], node);
        } else if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            ggml_xtc_run_flash_attn_kernels(fns[i], node);
        } else if (node->op == GGML_OP_RMS_NORM) {
            ggml_xtc_run_rms_norm_kernels(fns[i], node);
        } else if (node->op == GGML_OP_GLU) {
            ggml_xtc_run_swiglu_kernels(fns[i], node);
        } else if (node->op == GGML_OP_SET_ROWS) {
            ggml_xtc_run_set_rows_kernels(fns[i], node);
        } else {
            ggml_xtc_run_eltwise_kernel(*fns[i][0], node);
        }
    }

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i ggml_backend_xtc_i = {
    /* .get_name                = */ ggml_backend_xtc_get_name,
    /* .free                    = */ ggml_backend_xtc_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_xtc_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_xtc_guid(void) {
    static ggml_guid guid = { 0x58, 0x74, 0x63, 0x21, 0x9a, 0x4f, 0x6c, 0x2b, 0xb1, 0x07, 0x3d, 0x88, 0x51, 0x6e, 0x0c, 0x44 };
    return &guid;
}

ggml_backend_t ggml_backend_xtc_init(void) {
    ggml_backend_xtc_context * ctx = new ggml_backend_xtc_context;
    xtc_support_lib_init();

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_xtc_guid(),
        /* .iface   = */ ggml_backend_xtc_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xtc_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_xtc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_xtc_guid());
}

void ggml_backend_xtc_set_n_threads(ggml_backend_t backend_xtc, int n_threads) {
    GGML_ASSERT(ggml_backend_is_xtc(backend_xtc));

    ggml_backend_xtc_context * ctx = (ggml_backend_xtc_context *) backend_xtc->context;
    ctx->n_threads = n_threads;
}

// device interface

static const char * ggml_backend_xtc_device_get_name(ggml_backend_dev_t dev) {
    return "XTC";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_xtc_device_get_description(ggml_backend_dev_t dev) {
    return "xtc Transform Compiler";

    GGML_UNUSED(dev);
}

static void ggml_backend_xtc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_xtc_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_xtc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xtc_device_get_name(dev);
    props->description = ggml_backend_xtc_device_get_description(dev);
    props->type        = ggml_backend_xtc_device_get_type(dev);
    ggml_backend_xtc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_xtc_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_xtc_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_xtc_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_xtc_device_buffer_from_host_ptr(
        ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_xtc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    return ggml_xtc_device_supports_op(op);

    GGML_UNUSED(dev);
}

static bool ggml_backend_xtc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_xtc_device_i = {
    /* .get_name             = */ ggml_backend_xtc_device_get_name,
    /* .get_description      = */ ggml_backend_xtc_device_get_description,
    /* .get_memory           = */ ggml_backend_xtc_device_get_memory,
    /* .get_type             = */ ggml_backend_xtc_device_get_type,
    /* .get_props            = */ ggml_backend_xtc_device_get_props,
    /* .init_backend         = */ ggml_backend_xtc_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xtc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_xtc_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xtc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xtc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_xtc_reg_get_name(ggml_backend_reg_t reg) {
    return "XTC";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_xtc_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_xtc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_xtc_device = {
        /* .iface   = */ ggml_backend_xtc_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_xtc_device;

    GGML_UNUSED(index);
}

// Who runs the op: 0 = nobody (not claimed by the XTC device), 1 = XTC-compiled kernel,
// 2 = xtc_support_lib, 3 = free view op. Used by tests to report operator coverage.
// When an operator gets an XTC kernel, move it from the lib to the kernel classification.
static int ggml_backend_xtc_op_kind(const struct ggml_tensor * op) {
    return (int) ggml_xtc_classify_op(op);
}

static void * ggml_backend_xtc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) ggml_backend_xtc_set_n_threads;
    }
    if (strcmp(name, "ggml_backend_xtc_op_kind") == 0) {
        return (void *) ggml_backend_xtc_op_kind;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_xtc_reg_i = {
    /* .get_name         = */ ggml_backend_xtc_reg_get_name,
    /* .get_device_count = */ ggml_backend_xtc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xtc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_xtc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_xtc_reg(void) {
    static struct ggml_backend_reg ggml_backend_xtc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xtc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_xtc_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xtc_reg)
