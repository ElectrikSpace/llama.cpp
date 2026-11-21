#include "ggml-xtc.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "xtc_support_lib.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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
// Environment:
//   XTC_KERNELBASE           kernel base root (default: ./xtc_kernelbase)
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

//
// sha256 (ggml has none)
//

static std::string ggml_xtc_sha256_hex(const std::string & msg) {
    static const uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };

    std::vector<uint8_t> data(msg.begin(), msg.end());
    const uint64_t bit_len = (uint64_t) msg.size() * 8;
    data.push_back(0x80);
    while (data.size() % 64 != 56) {
        data.push_back(0);
    }
    for (int i = 7; i >= 0; i--) {
        data.push_back((uint8_t) (bit_len >> (8 * i)));
    }

    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };

    for (size_t off = 0; off < data.size(); off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            const uint8_t * p = &data[off + 4 * i];
            w[i] = ((uint32_t) p[0] << 24) | ((uint32_t) p[1] << 16) | ((uint32_t) p[2] << 8) | (uint32_t) p[3];
        }
        for (int i = 16; i < 64; i++) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            const uint32_t S1    = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch    = (e & f) ^ (~e & g);
            const uint32_t t1    = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0    = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj   = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2    = S0 + maj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    char out[65];
    for (int i = 0; i < 8; i++) {
        snprintf(out + 8 * i, 9, "%08x", h[i]);
    }
    return std::string(out, 64);
}

//
// matmul identity in the kernel base
//
// ggml MUL_MAT: dst[n,m] = sum_k src0[k,n] * src1[k,m] (ne0 contiguous), seen from XTC as
//   A = src1 : (M,K) row-major,  B = src0 : logical (K,N), physically (N,K) -> layout [1,0],
//   C = dst  : (M,N) row-major.
// This is xtc.kernelbase.ggml.ggml_matmul_graph(M, N, K, "float32").
//

// Kernel kinds. All of them are one XTC node on 2D row-major tensors, called with bare pointers (inputs..., output):
//   MATMUL         C[m,n] = A[m,k] x B[k,n], F32, B stored transposed (see above)
//   MATMUL_F16_BT  same with F16 A and B, F32 output (dtype attr)
//   MATMUL_F16     F16 A and B (B stored (k,n) as is), F32 output
//   ADD / MUL      C[m,n] = A[m,n] (+|*) B[m,n], F32
//   FMA            D[m,n] = A[m,n] * B[m,n] + C[m,n], F32
//   RSQRT / SILU   C[m,n] = 1/sqrt(A[m,n]) or A[m,n] / (1 + exp(-A[m,n])), F32
//   CAST_F16       C[m,n] = F16(A[m,n]), A F32 (round to nearest even)
enum ggml_xtc_kop {
    XTC_KOP_MATMUL, XTC_KOP_MATMUL_F16_BT, XTC_KOP_MATMUL_F16, XTC_KOP_ADD, XTC_KOP_MUL, XTC_KOP_FMA,
    XTC_KOP_RSQRT, XTC_KOP_SILU, XTC_KOP_CAST_F16,
};

struct ggml_xtc_kernel_shape {
    ggml_xtc_kop kop = XTC_KOP_MATMUL;
    int64_t m = 0, n = 0, k = 0; // k only for matmuls
    bool operator<(const ggml_xtc_kernel_shape & o) const {
        return std::tie(kop, m, n, k) < std::tie(o.kop, o.m, o.n, o.k);
    }
};

static const char * ggml_xtc_op_name(ggml_xtc_kop kop) {
    switch (kop) {
        case XTC_KOP_ADD: return "add";
        case XTC_KOP_MUL: return "mul";
        case XTC_KOP_FMA: return "fma";
        case XTC_KOP_RSQRT: return "rsqrt";
        case XTC_KOP_SILU: return "silu";
        case XTC_KOP_CAST_F16: return "cast";
        default:          return "matmul";
    }
}

static bool ggml_xtc_kop_is_matmul(ggml_xtc_kop kop) {
    return kop == XTC_KOP_MATMUL || kop == XTC_KOP_MATMUL_F16_BT || kop == XTC_KOP_MATMUL_F16;
}

struct ggml_xtc_input_desc {
    const char * dtype;
    int64_t      d0, d1;
    bool         transposed; // layout [1,0]
};

struct ggml_xtc_kernel_desc {
    std::vector<ggml_xtc_input_desc> inputs;
    const char *                     attr_dtype = nullptr; // dtype attribute (matmul accumulator, cast target)
};

static ggml_xtc_kernel_desc ggml_xtc_describe(const ggml_xtc_kernel_shape & s) {
    ggml_xtc_kernel_desc d;
    switch (s.kop) {
        case XTC_KOP_MATMUL:
            d.inputs = { { "float32", s.m, s.k, false }, { "float32", s.k, s.n, true } };
            break;
        case XTC_KOP_MATMUL_F16_BT:
            d.inputs = { { "float16", s.m, s.k, false }, { "float16", s.k, s.n, true } };
            d.attr_dtype = "float32";
            break;
        case XTC_KOP_MATMUL_F16:
            d.inputs = { { "float16", s.m, s.k, false }, { "float16", s.k, s.n, false } };
            d.attr_dtype = "float32";
            break;
        case XTC_KOP_ADD:
        case XTC_KOP_MUL:
            d.inputs = { { "float32", s.m, s.n, false }, { "float32", s.m, s.n, false } };
            break;
        case XTC_KOP_RSQRT:
        case XTC_KOP_SILU:
            d.inputs = { { "float32", s.m, s.n, false } };
            break;
        case XTC_KOP_CAST_F16:
            d.inputs = { { "float32", s.m, s.n, false } };
            d.attr_dtype = "float16";
            break;
        case XTC_KOP_FMA:
            d.inputs = { { "float32", s.m, s.n, false }, { "float32", s.m, s.n, false }, { "float32", s.m, s.n, false } };
            break;
    }
    return d;
}

static std::string ggml_xtc_shape_str(const ggml_xtc_kernel_shape & s) {
    char buf[128];
    if (ggml_xtc_kop_is_matmul(s.kop)) {
        snprintf(buf, sizeof(buf), "%s%s M=%" PRId64 " N=%" PRId64 " K=%" PRId64, ggml_xtc_op_name(s.kop),
                 s.kop == XTC_KOP_MATMUL ? " f32" : " f16->f32", s.m, s.n, s.k);
    } else {
        snprintf(buf, sizeof(buf), "%s f32%s M=%" PRId64 " N=%" PRId64, ggml_xtc_op_name(s.kop),
                 s.kop == XTC_KOP_CAST_F16 ? "->f16" : "", s.m, s.n);
    }
    return buf;
}

// keys sorted, no spaces: json.dumps(sort_keys=True, separators=(",", ":"))
static std::string ggml_xtc_canonical_json(const ggml_xtc_kernel_shape & s) {
    const ggml_xtc_kernel_desc d = ggml_xtc_describe(s);
    const int n_in = (int) d.inputs.size();
    std::string json = "{\"inputs\":[";
    for (int i = 0; i < n_in; i++) {
        const ggml_xtc_input_desc & in = d.inputs[i];
        char buf[256];
        snprintf(buf, sizeof(buf), "%s{\"expr\":{\"type\":{\"dtype\":\"%s\",%s\"shape\":[%" PRId64 ",%" PRId64 "]}},\"uid\":\"%%%d\"}",
                 i ? "," : "", in.dtype, in.transposed ? "\"layout\":[1,0]," : "", in.d0, in.d1, i);
        json += buf;
    }
    json += "],\"nodes\":[{\"expr\":{\"args\":[";
    for (int i = 0; i < n_in; i++) {
        json += (i ? ",\"%" : "\"%") + std::to_string(i) + "\"";
    }
    json += "],\"op\":{\"attrs\":";
    json += d.attr_dtype ? std::string("{\"dtype\":\"") + d.attr_dtype + "\"}" : "{}";
    json += ",\"name\":\"";
    json += ggml_xtc_op_name(s.kop);
    json += "\"}},\"uid\":\"%" + std::to_string(n_in) + "\"}],\"outputs\":[{\"uid\":\"%" + std::to_string(n_in) + "\"}]}";
    return json;
}

static std::string ggml_xtc_kernel_key(const ggml_xtc_kernel_shape & s) {
    return ggml_xtc_sha256_hex(ggml_xtc_canonical_json(s)).substr(0, 16);
}

// graph.yaml for the stub (json is valid yaml; same dict XTCGraph.to_dict() produces)
static std::string ggml_xtc_graph_yaml(const ggml_xtc_kernel_shape & s) {
    const ggml_xtc_kernel_desc d = ggml_xtc_describe(s);
    const std::string op = ggml_xtc_op_name(s.kop);
    std::string yaml = "name: " + op + "\nops_version: unknown\ninputs:\n";
    std::string args;
    for (size_t i = 0; i < d.inputs.size(); i++) {
        const ggml_xtc_input_desc & in = d.inputs[i];
        const std::string uid = op + "_in" + std::to_string(i);
        char buf[256];
        snprintf(buf, sizeof(buf), "- uid: %s\n  name: %c\n  expr:\n    type: {shape: [%" PRId64 ", %" PRId64 "], dtype: %s%s}\n",
                 uid.c_str(), (char) ('A' + i), in.d0, in.d1, in.dtype, in.transposed ? ", layout: [1, 0]" : "");
        yaml += buf;
        args += (i ? ", " : "") + uid;
    }
    yaml += "nodes:\n- uid: " + op + "_node0\n  name: C\n  expr:\n    op: {name: " + op + ", attrs: " +
            (d.attr_dtype ? std::string("{dtype: ") + d.attr_dtype + "}" : "{}") + "}\n    args: [" + args + "]\noutputs:\n- uid: " + op + "_node0\n";
    return yaml;
}

//
// kernel base access
//

using ggml_xtc_kernel_fn = void (*)(void *, void *, void *);

// KV cells per attention block (n_kv is a multiple of 256 in llama.cpp)
static const int64_t GGML_XTC_FA_BLOCK = 256;

enum class ggml_xtc_kernel_state { FOUND, MISSING, BROKEN };

struct ggml_xtc_kernel {
    ggml_xtc_kernel_state state = ggml_xtc_kernel_state::MISSING;
    ggml_xtc_kernel_fn    fn    = nullptr;
    std::string           key;
    std::string           error;   // when BROKEN
};

static std::mutex                                     g_xtc_mutex;
static std::map<ggml_xtc_kernel_shape, ggml_xtc_kernel> g_xtc_kernels;

static fs::path ggml_xtc_kernelbase_root() {
    const char * env = getenv("XTC_KERNELBASE");
    fs::path root = (env != nullptr && env[0] != '\0') ? fs::path(env) : fs::path("xtc_kernelbase");
    std::error_code ec;
    fs::path abs = fs::absolute(root, ec);
    return ec ? root : abs;
}

static std::string ggml_xtc_trim(std::string s) {
    const char * ws = " \t\r\n\"'";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) {
        return "";
    }
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

// minimal "top-level scalar key" reader for meta.yaml
static std::string ggml_xtc_meta_value(const fs::path & meta, const std::string & key) {
    std::ifstream in(meta);
    std::string line;
    const std::string prefix = key + ":";
    while (std::getline(in, line)) {
        if (line.compare(0, prefix.size(), prefix) == 0) {
            return ggml_xtc_trim(line.substr(prefix.size()));
        }
    }
    return "";
}

static void * ggml_xtc_dlopen(const char * path) {
#if defined(_WIN32)
    return (void *) LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void * ggml_xtc_dlsym(void * handle, const char * sym) {
#if defined(_WIN32)
    return (void *) GetProcAddress((HMODULE) handle, sym);
#else
    return dlsym(handle, sym);
#endif
}

static std::string ggml_xtc_dlerror() {
#if defined(_WIN32)
    return "LoadLibrary failed";
#else
    const char * e = dlerror();
    return e ? e : "unknown error";
#endif
}

// g_xtc_mutex must be held
static ggml_xtc_kernel & ggml_xtc_get_kernel(const ggml_xtc_kernel_shape & shape) {
    auto it = g_xtc_kernels.find(shape);
    if (it != g_xtc_kernels.end()) {
        return it->second;
    }

    ggml_xtc_kernel k;
    k.key = ggml_xtc_kernel_key(shape);

    const fs::path dir  = ggml_xtc_kernelbase_root() / k.key;
    const fs::path meta = dir / "meta.yaml";
    const fs::path so   = dir / "kernel.so";

    if (fs::exists(meta) && fs::exists(so)) {
        std::string sym = ggml_xtc_meta_value(meta, "payload_name");
        if (sym.empty()) {
            sym = ggml_xtc_op_name(shape.kop);
        }
        void * handle = ggml_xtc_dlopen(so.string().c_str());
        if (handle == nullptr) {
            k.state = ggml_xtc_kernel_state::BROKEN;
            k.error = "cannot load " + so.string() + ": " + ggml_xtc_dlerror();
        } else if (void * fn = ggml_xtc_dlsym(handle, sym.c_str())) {
            k.state = ggml_xtc_kernel_state::FOUND;
            k.fn    = (ggml_xtc_kernel_fn) fn;
        } else {
            k.state = ggml_xtc_kernel_state::BROKEN;
            k.error = "symbol '" + sym + "' not found in " + so.string();
        }
    }

    return g_xtc_kernels.emplace(shape, std::move(k)).first->second;
}

// Supported: F32, contiguous, src0 is a model weight. ne2/ne3 are handled by a batch loop around
// the kernel; M (= number of tokens in the batch) is a static shape of the kernel.
static bool ggml_xtc_is_supported_matmul(const struct ggml_tensor * op, ggml_xtc_kernel_shape * shape) {
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    // only matmuls with a model weight: attention matmuls over the KV cache have a shape
    // that depends on the context length, which static-shape kernels can not follow
    if (src0->buffer == nullptr || ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (ggml_nelements(op) == 0 || ggml_nelements(src0) == 0 || ggml_nelements(src1) == 0) {
        return false;
    }
    if (src1->ne[2] % src0->ne[2] != 0 || src1->ne[3] % src0->ne[3] != 0) {
        return false;
    }
    if (shape != nullptr) {
        shape->kop = XTC_KOP_MATMUL;
        shape->m = src1->ne[1];
        shape->n = src0->ne[1];
        shape->k = src0->ne[0];
    }
    return true;
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
            // Q.K^T and P.V on blocks of GGML_XTC_FA_BLOCK KV cells, the online softmax stays in C
            if (ggml_xtc_lib_supports_op(op) && src1->ne[1] % GGML_XTC_FA_BLOCK == 0) {
                const int64_t M = (src0->ne[2] / src1->ne[2]) * src0->ne[1];
                const int64_t D = src0->ne[0];
                out.push_back({ XTC_KOP_MATMUL_F16_BT, M, GGML_XTC_FA_BLOCK, D });
                out.push_back({ XTC_KOP_MATMUL_F16, M, D, GGML_XTC_FA_BLOCK });
            }
            break;
        default:
            break;
    }
    return out;
}

//
// stubs
//

struct ggml_xtc_missing_entry {
    std::string           key;
    ggml_xtc_kernel_shape shape;
    std::string           src0_name;
    std::string           stub_dir;
    bool                  created = false;
    bool                  existed = false;
};

static bool ggml_xtc_write_file(const fs::path & path, const std::string & content) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out << content;
    return (bool) out;
}

static const char * GGML_XTC_STUB_TEMPLATE = R"PY(#!/usr/bin/env python3
"""XTC kernel stub generated by ggml-xtc (llama.cpp).

    op     : @OP@
    shape  : @SHAPE@
    key    : @KEY@
    source : @WEIGHT@

All that is left to do is to provide the schedule: fill SPEC below and set TODO = False. Then

    python @SCRIPT@

compiles the kernel, validates it against the reference and inserts it in the kernel base
(@ROOT@). Then rerun llama.cpp with the same model and arguments.
(graph.yaml, next to this file, is the exact subgraph llama.cpp asks for: do not edit it.)
"""
from xtc.kernelbase.pending import run_stub

TODO = True  # <- set to False once SPEC is yours

# descript syntax: xtc/src/xtc/schedules/descript.py
# examples:        xtc/tests/filecheck/schedules/test_matmul_descript_mlir.py
NODE_NAME = "C"
ABSTRACT_DIMS = @DIMS@  # i: rows of C (M), j: columns of C (N)@KDIM@

# TODO: your schedule. The baseline below (loops left as they are) is valid but slow.
# Example of a tiled one (tile sizes must be compatible with the dims):
#   SPEC = {"i": {}, "j": {}, "j#8": {"vectorize": True}}
SPEC = @SPEC@

if __name__ == "__main__":
    raise SystemExit(run_stub(__file__, NODE_NAME, ABSTRACT_DIMS, SPEC, todo=TODO))
)PY";

static std::string ggml_xtc_replace_all(std::string s, const std::string & from, const std::string & to) {
    for (size_t pos = 0; (pos = s.find(from, pos)) != std::string::npos; pos += to.size()) {
        s.replace(pos, from.size(), to);
    }
    return s;
}

static void ggml_xtc_write_stub(const fs::path & root, ggml_xtc_missing_entry & e) {
    const fs::path dir    = root / "_pending" / e.key;
    const fs::path script = dir / "build_kernel.py";
    e.stub_dir = dir.string();

    std::error_code ec;
    if (fs::exists(script)) {
        // never overwrite the user's work in progress
        e.existed = true;
        return;
    }
    fs::create_directories(dir, ec);
    if (ec) {
        return;
    }

    std::string py = GGML_XTC_STUB_TEMPLATE;
    const bool matmul = ggml_xtc_kop_is_matmul(e.shape.kop);
    py = ggml_xtc_replace_all(py, "@OP@",     ggml_xtc_op_name(e.shape.kop));
    py = ggml_xtc_replace_all(py, "@SHAPE@",  ggml_xtc_shape_str(e.shape));
    py = ggml_xtc_replace_all(py, "@DIMS@",   matmul ? "[\"i\", \"j\", \"k\"]" : "[\"i\", \"j\"]");
    py = ggml_xtc_replace_all(py, "@KDIM@",   matmul ? ", k: reduction (K)" : "");
    py = ggml_xtc_replace_all(py, "@SPEC@",   matmul ? "{\n    \"i\": {},\n    \"j\": {},\n    \"k\": {},\n}" :
                                                       "{\n    \"i\": {},\n    \"j\": {},\n}");
    py = ggml_xtc_replace_all(py, "@KEY@",    e.key);
    py = ggml_xtc_replace_all(py, "@WEIGHT@", e.src0_name);
    py = ggml_xtc_replace_all(py, "@SCRIPT@", script.string());
    py = ggml_xtc_replace_all(py, "@ROOT@",   root.string());

    const bool ok = ggml_xtc_write_file(dir / "graph.yaml", ggml_xtc_graph_yaml(e.shape)) &&
                    ggml_xtc_write_file(script, py);
    if (ok) {
        fs::permissions(script, fs::perms::owner_exec | fs::perms::group_exec | fs::perms::others_exec,
                        fs::perm_options::add, ec);
    }
    e.created = ok;
}

static std::string ggml_xtc_failure_message(const fs::path & root,
                                            const std::vector<ggml_xtc_missing_entry> & missing,
                                            const std::vector<std::string> & broken) {
    std::string msg;
    char line[1024];

    if (!missing.empty()) {
        snprintf(line, sizeof(line),
            "\nggml-xtc: %zu XTC kernel(s) missing from the kernel base '%s'\n",
            missing.size(), root.string().c_str());
        msg += line;
        for (const auto & e : missing) {
            snprintf(line, sizeof(line), "  - %s  %s  (%s)\n      stub: %s%s\n",
                e.key.c_str(), ggml_xtc_shape_str(e.shape).c_str(), e.src0_name.c_str(), e.stub_dir.c_str(),
                e.created ? "" : (e.existed ? "  (already exists, kept as is)" : "  (could NOT be written!)"));
            msg += line;
        }
        snprintf(line, sizeof(line),
            "To fix it:\n"
            "  1. in each <stub>/build_kernel.py, fill SPEC (the XTC schedule) and set TODO = False\n"
            "  2. python <stub>/build_kernel.py     (or all at once: xtc-kernelbase build --root '%s')\n"
            "     -> compiles, validates and inserts the kernel in the kernel base\n"
            "  3. rerun llama.cpp with the same model and arguments\n"
            "Tips: kernels have a static shape and M is the number of tokens in the micro-batch, so each\n"
            "      micro-batch size needs its own kernels. Run with `-ub 1` to only need the M=1 kernels.\n"
            "      GGML_XTC_MISSING_KERNEL=cpu leaves those matmuls to the CPU backend instead of failing.\n",
            root.string().c_str());
        msg += line;
    }
    for (const auto & b : broken) {
        msg += "ggml-xtc: broken kernel in the kernel base: " + b + "\n";
    }
    return msg;
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

static void ggml_xtc_run_mul_mat_kernel(ggml_xtc_kernel_fn fn, struct ggml_tensor * dst) {
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

            // base addresses of the three 2D operands (byte strides, as in ggml)
            void * a = (void *) ((char *) src1->data + i12 * nb12 + i13 * nb13);
            void * b = (void *) ((char *) src0->data + i02 * nb02 + i03 * nb03);
            void * c = (void *) ((char *) dst->data  + i12 * nb2  + i13 * nb3);

            fn(a, b, c);
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

using ggml_xtc_kernel_fn4 = void (*)(void *, void *, void *, void *);

// one input, one output
static void ggml_xtc_call_unary(ggml_xtc_kernel_fn fn, void * in, void * out) {
    ((void (*)(void *, void *)) fn)(in, out);
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
static void ggml_xtc_run_rms_norm_kernels(const std::vector<ggml_xtc_kernel_fn> & fns, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t N = dst->ne[0];
    const int64_t M = ggml_xtc_flat_rows(dst) ? ggml_nrows(dst) : 1;
    std::vector<float> inv_n(N, 1.0f / (float) N), ones(N, 1.0f), eps_v(M, eps);
    std::vector<float> sq(M * N), bc(M * N), mean(M), mean_eps(M), rs(M);

    ggml_xtc_for_each_chunk(dst, [&](int64_t i1, int64_t i2, int64_t i3) {
        void * x = ggml_xtc_row_ptr(src0, i1, i2, i3);
        fns[0](x, x, sq.data());
        fns[1](sq.data(), inv_n.data(), mean.data());
        fns[2](mean.data(), eps_v.data(), mean_eps.data());
        ggml_xtc_call_unary(fns[3], mean_eps.data(), rs.data());
        fns[4](rs.data(), ones.data(), bc.data());
        fns[5](x, bc.data(), ggml_xtc_row_ptr(dst, i1, i2, i3));
    });
}

// kernels: silu(gate), then * up
static void ggml_xtc_run_swiglu_kernels(const std::vector<ggml_xtc_kernel_fn> & fns, struct ggml_tensor * dst) {
    const struct ggml_tensor * gate = dst->src[0];
    const struct ggml_tensor * up   = dst->src[1];
    const int64_t M = ggml_xtc_flat_rows(dst) ? ggml_nrows(dst) : 1;
    std::vector<float> tmp(M * dst->ne[0]);

    ggml_xtc_for_each_chunk(dst, [&](int64_t i1, int64_t i2, int64_t i3) {
        ggml_xtc_call_unary(fns[0], ggml_xtc_row_ptr(gate, i1, i2, i3), tmp.data());
        fns[1](tmp.data(), ggml_xtc_row_ptr(up, i1, i2, i3), ggml_xtc_row_ptr(dst, i1, i2, i3));
    });
}

// kernel: F32 -> F16 cast of one row into the destination row
static void ggml_xtc_run_set_rows_kernels(const std::vector<ggml_xtc_kernel_fn> & fns, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * idx  = dst->src[1];
    for (int64_t i3 = 0; i3 < src0->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < src0->ne[2]; i2++) {
            for (int64_t i = 0; i < src0->ne[1]; i++) {
                const int64_t row = *(const int64_t *) ((const char *) idx->data + i * idx->nb[0] +
                                                        (i2 % idx->ne[1]) * idx->nb[1] + (i3 % idx->ne[2]) * idx->nb[2]);
                ggml_xtc_call_unary(fns[0], ggml_xtc_row_ptr(src0, i, i2, i3), ggml_xtc_row_ptr(dst, row, i2, i3));
            }
        }
    }
}

// one call of the (1, N) or (M, N) kernel on the whole tensor when it is flat, else one call per row of dst
static void ggml_xtc_run_eltwise_kernel(ggml_xtc_kernel_fn fn, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (ggml_xtc_is_flat_eltwise(dst)) {
        fn(src0->data, src1->data, dst->data);
        return;
    }
    for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
                void * a = (char *) src0->data + i1 * src0->nb[1] + i2 * src0->nb[2] + i3 * src0->nb[3];
                void * b = (char *) src1->data + (i1 % src1->ne[1]) * src1->nb[1] + (i2 % src1->ne[2]) * src1->nb[2] +
                           (i3 % src1->ne[3]) * src1->nb[3];
                void * d = (char *) dst->data + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3];
                fn(a, b, d);
            }
        }
    }
}

// kernels: matmul by the swap matrix, mul by sin, fma with cos (see ggml_xtc_node_kernels)
static void ggml_xtc_run_rope_kernels(const std::vector<ggml_xtc_kernel_fn> & fns, struct ggml_tensor * dst) {
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

    for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
            xtc_rope_factors(&p, D, pos[i2], ff, c.data(), s.data());
            for (int64_t h = 0; h < H; h++) {
                memcpy(&C[h * D], c.data(), D * sizeof(float));
                memcpy(&S[h * D], s.data(), D * sizeof(float));
            }
            void * x   = (char *) src0->data + i2 * src0->nb[2] + i3 * src0->nb[3];
            void * out = (char *) dst->data + i2 * dst->nb[2] + i3 * dst->nb[3];
            fns[0](x, swap.data(), t1.data());
            fns[1](t1.data(), S.data(), t2.data());
            ((ggml_xtc_kernel_fn4) fns[2])(x, C.data(), t2.data(), out);
        }
    }
}

// Per (sequence, KV head): the queries of the GQA group are the rows of one matmul. For each block of
// GGML_XTC_FA_BLOCK cells: S = Q.K^T (kernel), online softmax in C, then acc += P.V (kernel).
// Like the CPU backend, Q is rounded to F16, and here the weights P too.
static void ggml_xtc_run_flash_attn_kernels(const std::vector<ggml_xtc_kernel_fn> & fns, struct ggml_tensor * dst) {
    const struct ggml_tensor * q    = dst->src[0];
    const struct ggml_tensor * k    = dst->src[1];
    const struct ggml_tensor * v    = dst->src[2];
    const struct ggml_tensor * mask = dst->src[3];
    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const int64_t D      = q->ne[0];
    const int64_t n_tok  = q->ne[1];
    const int64_t n_kv   = k->ne[1];
    const int64_t rk2    = q->ne[2] / k->ne[2];
    const int64_t M      = rk2 * n_tok;
    const int64_t BL     = GGML_XTC_FA_BLOCK;

    std::vector<ggml_fp16_t> qh(M * D), kb(BL * D), vb(BL * D), P(M * BL);
    std::vector<float>       S(M * BL), O(M * D), acc(M * D), m(M), l(M), sc(BL);

    for (int64_t s = 0; s < q->ne[3]; s++) {
        for (int64_t g = 0; g < k->ne[2]; g++) {
            for (int64_t hh = 0; hh < rk2; hh++) {
                for (int64_t t = 0; t < n_tok; t++) {
                    const float * qp = (const float *) ((const char *) q->data + t * q->nb[1] + (g * rk2 + hh) * q->nb[2] + s * q->nb[3]);
                    for (int64_t d = 0; d < D; d++) {
                        qh[(hh * n_tok + t) * D + d] = ggml_fp32_to_fp16(qp[d]);
                    }
                }
            }
            std::fill(acc.begin(), acc.end(), 0.0f);
            std::fill(m.begin(), m.end(), -INFINITY);
            std::fill(l.begin(), l.end(), 0.0f);

            for (int64_t b0 = 0; b0 < n_kv; b0 += BL) {
                for (int64_t j = 0; j < BL; j++) {
                    memcpy(&kb[j * D], (const char *) k->data + (b0 + j) * k->nb[1] + g * k->nb[2] + s * k->nb[3], D * sizeof(ggml_fp16_t));
                }
                fns[0](qh.data(), kb.data(), S.data());

                bool any = false;
                for (int64_t r = 0; r < M; r++) {
                    const int64_t h = g * rk2 + r / n_tok;
                    const int64_t t = r % n_tok;
                    const ggml_fp16_t * mp = mask ? (const ggml_fp16_t *) ((const char *) mask->data + t * mask->nb[1] +
                        (h % mask->ne[2]) * mask->nb[2] + (s % mask->ne[3]) * mask->nb[3]) + b0 : nullptr;

                    float mb = -INFINITY;
                    for (int64_t j = 0; j < BL; j++) {
                        sc[j] = S[r * BL + j] * scale + (mp ? ggml_fp16_to_fp32(mp[j]) : 0.0f);
                        mb    = std::max(mb, sc[j]);
                    }
                    ggml_fp16_t * Pr = &P[r * BL];
                    if (mb == -INFINITY) {
                        std::fill(Pr, Pr + BL, ggml_fp32_to_fp16(0.0f));
                        continue;
                    }
                    any = true;
                    const float m_new = std::max(m[r], mb);
                    const float rs    = expf(m[r] - m_new); // 0 on the first block (m = -inf)
                    for (int64_t d = 0; d < D; d++) {
                        acc[r * D + d] *= rs;
                    }
                    l[r] *= rs;
                    for (int64_t j = 0; j < BL; j++) {
                        Pr[j] = ggml_fp32_to_fp16(expf(sc[j] - m_new));
                        l[r] += ggml_fp16_to_fp32(Pr[j]);
                    }
                    m[r] = m_new;
                }
                if (!any) {
                    continue;
                }
                for (int64_t j = 0; j < BL; j++) {
                    memcpy(&vb[j * D], (const char *) v->data + (b0 + j) * v->nb[1] + g * v->nb[2] + s * v->nb[3], D * sizeof(ggml_fp16_t));
                }
                fns[1](P.data(), vb.data(), O.data());
                for (int64_t i = 0; i < M * D; i++) {
                    acc[i] += O[i];
                }
            }

            for (int64_t r = 0; r < M; r++) {
                const int64_t h = g * rk2 + r / n_tok;
                const int64_t t = r % n_tok;
                float *       out = (float *) ((char *) dst->data + h * dst->nb[1] + t * dst->nb[2] + s * dst->nb[3]);
                const float   inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
                for (int64_t d = 0; d < D; d++) {
                    out[d] = acc[r * D + d] * inv;
                }
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
    std::vector<std::vector<ggml_xtc_kernel_fn>> fns(cgraph->n_nodes);
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
                        fns[i].push_back(k.fn);
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
            ggml_xtc_run_mul_mat_kernel(fns[i][0], node);
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
            ggml_xtc_run_eltwise_kernel(fns[i][0], node);
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
