#pragma once

// Host independent part of the XTC backends: kernel base keys, kernel loading and the stubs
// written for missing kernels.

#include "ggml.h"
#include "ggml-backend.h"
#include "xtc_kernel_abi.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

// Kernel kinds. All of them are one XTC node on 2D row-major tensors, called with bare pointers (inputs..., output):
//   MATMUL         C[m,n] = A[m,k] x B[k,n], F32, B stored transposed (see ggml_xtc_is_supported_matmul)
//   MATMUL_KN      same as MATMUL with B stored (k,n) as is: XTC_MPPA keeps the F32 model weights transposed on the device
//                  (see ggml_xtc_mppa_weight_kn), which lets the 8x8 microkernel run on them
//   MATMUL_F16_BT  same with F16 A and B, F32 output (dtype attr)
//   MATMUL_F16     F16 A and B (B stored (k,n) as is), F32 output
//   ADD / MUL      C[m,n] = A[m,n] (+|*) B[m,n], F32
//   FMA            D[m,n] = A[m,n] * B[m,n] + C[m,n], F32
//   RSQRT / SILU / EXP   C[m,n] = 1/sqrt(A[m,n]), A[m,n] / (1 + exp(-A[m,n])) or exp(A[m,n]), F32
//   REDUCE_MAX / SUM     C[m] = max or sum over n of A[m,n], F32
//   SUB_ROW / DIV_ROW    C[m,n] = A[m,n] (-|/) B[m], F32
//   CAST_F16       C[m,n] = F16(A[m,n]), A F32 (round to nearest even)
enum ggml_xtc_kop {
    XTC_KOP_MATMUL, XTC_KOP_MATMUL_KN, XTC_KOP_MATMUL_F16_BT, XTC_KOP_MATMUL_F16, XTC_KOP_ADD, XTC_KOP_MUL, XTC_KOP_FMA,
    XTC_KOP_RSQRT, XTC_KOP_SILU, XTC_KOP_CAST_F16, XTC_KOP_EXP,
    XTC_KOP_REDUCE_MAX, XTC_KOP_REDUCE_SUM, XTC_KOP_SUB_ROW, XTC_KOP_DIV_ROW,
};

// f16: F16 variant of the kernel: all inputs and the output are F16 (the MATMUL_F16* kinds get an F16 output, CAST_F16 is
// an F16 -> F16 copy). Used by XTC_MPPA only (ggml_xtc_node_kernels_f16).
struct ggml_xtc_kernel_shape {
    ggml_xtc_kop kop = XTC_KOP_MATMUL;
    int64_t m = 0, n = 0, k = 0; // k only for matmuls
    bool f16 = false;
    bool operator<(const ggml_xtc_kernel_shape & o) const {
        return std::tie(kop, m, n, k, f16) < std::tie(o.kop, o.m, o.n, o.k, o.f16);
    }
};

struct ggml_xtc_input_desc {
    const char * dtype;
    int64_t      d0, d1; // d1 = 0: 1D tensor (d0)
    bool         transposed; // layout [1,0]
};

struct ggml_xtc_kernel_desc {
    std::vector<ggml_xtc_input_desc> inputs;
    const char *                     attr_dtype = nullptr; // dtype attribute (matmul accumulator, cast target)
};

const char *             ggml_xtc_op_name(ggml_xtc_kop kop);
bool                     ggml_xtc_kop_is_matmul(ggml_xtc_kop kop);
ggml_xtc_kernel_desc     ggml_xtc_describe(const ggml_xtc_kernel_shape & s);
std::string              ggml_xtc_shape_str(const ggml_xtc_kernel_shape & s);
std::string ggml_xtc_kernel_key(const ggml_xtc_kernel_shape & s, const std::string & target = "");
std::string              ggml_xtc_graph_yaml(const ggml_xtc_kernel_shape & s);

using ggml_xtc_kernel_fn = void (*)(void *, void *, void *); // raw kernel.so symbol: cast by the number of inputs

enum class ggml_xtc_kernel_state { FOUND, MISSING, BROKEN };

struct ggml_xtc_kernel {
    ggml_xtc_kernel_state state = ggml_xtc_kernel_state::MISSING;
    xtc_kernel_call_fn    call  = nullptr; // bundle wrapper (checked)
    ggml_xtc_kernel_fn    fn    = nullptr; // kernel.so symbol (unchecked)
    int                   n_inputs = 0;
    std::string           key;
    std::string           error;   // when BROKEN
};

// protects the kernel tables below: lock it around ggml_xtc_get_kernel()
extern std::mutex g_xtc_mutex;

// g_xtc_mutex must be held
ggml_xtc_kernel & ggml_xtc_get_kernel(const ggml_xtc_kernel_shape & shape);

std::filesystem::path ggml_xtc_kernelbase_root();
bool                  ggml_xtc_bundle_only();

// Supported: F32, contiguous, src0 is a model weight. ne2/ne3 are handled by a batch loop around
// the kernel; M (= number of tokens in the batch) is a static shape of the kernel.
bool ggml_xtc_is_supported_matmul(const struct ggml_tensor * op, ggml_xtc_kernel_shape * shape);

// F32 ADD / MUL whose inputs and output are contiguous and of the same shape: one kernel call on (M, N)
bool ggml_xtc_is_flat_eltwise(const struct ggml_tensor * op);

// all the rows of the node and of its sources are packed, so one kernel call can take them all
bool ggml_xtc_flat_rows(const struct ggml_tensor * op);

// Kernels needed by a node, in call order. Empty when xtc_support_lib runs the node.
std::vector<ggml_xtc_kernel_shape> ggml_xtc_node_kernels(const struct ggml_tensor * op);

// Same for the F16 nodes of XTC_MPPA (F16 activations and weights; empty for any other node):
//   MUL_MAT (F16 weight, F16 input, F32 output: F16 x F16 -> F32 kernel), ADD / MUL, RMS_NORM, SWIGLU, ROPE, SET_ROWS:
//   the same kernels as ggml_xtc_node_kernels with the f16 flag; FLASH_ATTN_EXT with an F16 Q (F32 output, as ggml
//   defines it): no Q cast, F16 scores; CPY F32 -> F16: one cast kernel.
std::vector<ggml_xtc_kernel_shape> ggml_xtc_node_kernels_f16(const struct ggml_tensor * op);

// F16 tensor with packed rows (nb[0] == sizeof(ggml_fp16_t)), false for null
bool ggml_xtc_f16_rows(const struct ggml_tensor * t);

// ADD / MUL whose inputs and output are contiguous F16 tensors of the same shape
bool ggml_xtc_is_flat_eltwise_f16(const struct ggml_tensor * op);

// F32 tensor with packed rows (nb[0] == sizeof(float)), false for null
bool ggml_xtc_f32_rows(const struct ggml_tensor * t);

// xtc_support_lib can run the node on the host: op params and shapes are valid for its reference kernels
bool ggml_xtc_lib_supports_op(const struct ggml_tensor * op);

// value of "key: value" in a meta.yaml file, empty when absent
std::string ggml_xtc_meta_value(const std::filesystem::path & meta, const std::string & key);

struct ggml_xtc_missing_entry {
    std::string           key;
    ggml_xtc_kernel_shape shape;
    std::string           src0_name;
    std::string           stub_dir;
    bool                  created = false;
    bool                  existed = false;
};

void        ggml_xtc_write_stub(const std::filesystem::path & root, ggml_xtc_missing_entry & e);
std::string ggml_xtc_failure_message(const std::filesystem::path & root,
                                     const std::vector<ggml_xtc_missing_entry> & missing,
                                     const std::vector<std::string> & broken);
