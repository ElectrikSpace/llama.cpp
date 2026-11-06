#include "ggml-xtc.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

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
// Every supported kernel (only F32 MUL_MAT for now) is looked up in the XTC kernel base
// (see xtc/src/xtc/kernelbase). The key is the first 16 hex digits of the sha256 of the
// canonical XTC subgraph, which this file rebuilds as a JSON string. It must stay
// byte-identical to xtc.kernelbase.canonical.canonical_graph_json() for ggml_matmul_graph().
//
// If a kernel is missing, graph_compute fails *before computing anything*, and a stub
// (graph.yaml + a build_kernel.py where only the schedule is missing) is written to <kernelbase>/_pending/<key>/.
// The user fills the schedule in build_kernel.py, runs it, then reruns llama.cpp.
//
// Environment:
//   XTC_KERNELBASE           kernel base root (default: ./xtc_kernelbase)
//   GGML_XTC_MISSING_KERNEL  fail (default) | cpu   (cpu: silently fall back to the CPU backend)
//   GGML_XTC_DISABLE         1 -> the XTC device claims no op at all
//

struct ggml_backend_xtc_context {
    ggml_backend_t cpu = nullptr;
    int            n_threads = GGML_DEFAULT_N_THREADS;
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

struct ggml_xtc_matmul_shape {
    int64_t m, n, k;
    bool operator<(const ggml_xtc_matmul_shape & o) const {
        return std::tie(m, n, k) < std::tie(o.m, o.n, o.k);
    }
};

static std::string ggml_xtc_matmul_canonical_json(const ggml_xtc_matmul_shape & s) {
    // keys sorted, no spaces: json.dumps(sort_keys=True, separators=(",", ":"))
    char buf[512];
    snprintf(buf, sizeof(buf),
        "{\"inputs\":["
          "{\"expr\":{\"type\":{\"dtype\":\"float32\",\"shape\":[%" PRId64 ",%" PRId64 "]}},\"uid\":\"%%0\"},"
          "{\"expr\":{\"type\":{\"dtype\":\"float32\",\"layout\":[1,0],\"shape\":[%" PRId64 ",%" PRId64 "]}},\"uid\":\"%%1\"}],"
         "\"nodes\":[{\"expr\":{\"args\":[\"%%0\",\"%%1\"],\"op\":{\"attrs\":{},\"name\":\"matmul\"}},\"uid\":\"%%2\"}],"
         "\"outputs\":[{\"uid\":\"%%2\"}]}",
        s.m, s.k, s.k, s.n);
    return buf;
}

static std::string ggml_xtc_matmul_key(const ggml_xtc_matmul_shape & s) {
    return ggml_xtc_sha256_hex(ggml_xtc_matmul_canonical_json(s)).substr(0, 16);
}

// graph.yaml for the stub (json is valid yaml; same dict XTCGraph.to_dict() produces)
static std::string ggml_xtc_matmul_graph_yaml(const ggml_xtc_matmul_shape & s) {
    char buf[1024];
    snprintf(buf, sizeof(buf),
        "name: matmul\n"
        "ops_version: unknown\n"
        "inputs:\n"
        "- uid: matmul_in0\n"
        "  name: A\n"
        "  expr:\n"
        "    type: {shape: [%" PRId64 ", %" PRId64 "], dtype: float32}\n"
        "- uid: matmul_in1\n"
        "  name: B\n"
        "  expr:\n"
        "    type: {shape: [%" PRId64 ", %" PRId64 "], dtype: float32, layout: [1, 0]}\n"
        "nodes:\n"
        "- uid: matmul_node0\n"
        "  name: C\n"
        "  expr:\n"
        "    op: {name: matmul, attrs: {}}\n"
        "    args: [matmul_in0, matmul_in1]\n"
        "outputs:\n"
        "- uid: matmul_node0\n",
        s.m, s.k, s.k, s.n);
    return buf;
}

//
// kernel base access
//

using ggml_xtc_kernel_fn = void (*)(void *, void *, void *);

enum class ggml_xtc_kernel_state { FOUND, MISSING, BROKEN };

struct ggml_xtc_kernel {
    ggml_xtc_kernel_state state = ggml_xtc_kernel_state::MISSING;
    ggml_xtc_kernel_fn    fn    = nullptr;
    std::string           key;
    std::string           error;   // when BROKEN
};

static std::mutex                                     g_xtc_mutex;
static std::map<ggml_xtc_matmul_shape, ggml_xtc_kernel> g_xtc_kernels;

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
static ggml_xtc_kernel & ggml_xtc_get_matmul_kernel(const ggml_xtc_matmul_shape & shape) {
    auto it = g_xtc_kernels.find(shape);
    if (it != g_xtc_kernels.end()) {
        return it->second;
    }

    ggml_xtc_kernel k;
    k.key = ggml_xtc_matmul_key(shape);

    const fs::path dir  = ggml_xtc_kernelbase_root() / k.key;
    const fs::path meta = dir / "meta.yaml";
    const fs::path so   = dir / "kernel.so";

    if (fs::exists(meta) && fs::exists(so)) {
        std::string sym = ggml_xtc_meta_value(meta, "payload_name");
        if (sym.empty()) {
            sym = "matmul";
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
static bool ggml_xtc_is_supported_matmul(const struct ggml_tensor * op, ggml_xtc_matmul_shape * shape) {
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
        shape->m = src1->ne[1];
        shape->n = src0->ne[1];
        shape->k = src0->ne[0];
    }
    return true;
}

//
// stubs
//

struct ggml_xtc_missing_entry {
    std::string           key;
    ggml_xtc_matmul_shape shape;
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

    op     : matmul (ggml MUL_MAT, f32)
    shape  : M=@M@ N=@N@ K=@K@   (C[M,N] = A[M,K] x B[K,N], B stored transposed)
    key    : @KEY@
    weight : @WEIGHT@

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
ABSTRACT_DIMS = ["i", "j", "k"]  # i: rows of C (M), j: columns of C (N), k: reduction (K)

# TODO: your schedule. The baseline below (loops left as they are) is valid but slow.
# Example of a tiled one (tile sizes must be compatible with the dims):
#   SPEC = {"k": {}, "i": {}, "j": {}, "k#8": {"unroll": True}, "j#8": {"vectorize": True}}
SPEC = {
    "i": {},
    "j": {},
    "k": {},
}

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
    py = ggml_xtc_replace_all(py, "@M@",      std::to_string(e.shape.m));
    py = ggml_xtc_replace_all(py, "@N@",      std::to_string(e.shape.n));
    py = ggml_xtc_replace_all(py, "@K@",      std::to_string(e.shape.k));
    py = ggml_xtc_replace_all(py, "@KEY@",    e.key);
    py = ggml_xtc_replace_all(py, "@WEIGHT@", e.src0_name);
    py = ggml_xtc_replace_all(py, "@SCRIPT@", script.string());
    py = ggml_xtc_replace_all(py, "@ROOT@",   root.string());

    const bool ok = ggml_xtc_write_file(dir / "graph.yaml", ggml_xtc_matmul_graph_yaml(e.shape)) &&
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
            snprintf(line, sizeof(line),
                "  - %s  matmul f32 M=%" PRId64 " N=%" PRId64 " K=%" PRId64 "  (%s)\n      stub: %s%s\n",
                e.key.c_str(), e.shape.m, e.shape.n, e.shape.k, e.src0_name.c_str(), e.stub_dir.c_str(),
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
            "      GGML_XTC_MISSING_KERNEL=cpu falls back to the CPU backend instead of failing.\n",
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

static struct ggml_cgraph ggml_xtc_graph_view(struct ggml_cgraph * cgraph, int i0, int i1) {
    struct ggml_cgraph view = {
        /*.size             =*/ 0,
        /*.n_nodes          =*/ i1 - i0,
        /*.n_leafs          =*/ 0,
        /*.nodes            =*/ cgraph->nodes + i0,
        /*.grads            =*/ NULL,
        /*.grad_accs        =*/ NULL,
        /*.leafs            =*/ NULL,
        /*.use_counts       =*/ cgraph->use_counts,
        /*.visited_hash_set =*/ cgraph->visited_hash_set,
        /*.order            =*/ cgraph->order,
        /*.uid              =*/ 0,
    };
    return view;
}

static void ggml_xtc_run_mul_mat_kernel(ggml_xtc_kernel_fn fn, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;

            void * a = (void *) ((char *) src1->data + i12 * nb12 + i13 * nb13);
            void * b = (void *) ((char *) src0->data + i02 * nb02 + i03 * nb03);
            void * c = (void *) ((char *) dst->data  + i12 * nb2  + i13 * nb3);

            fn(a, b, c);
        }
    }
}

static enum ggml_status ggml_xtc_flush_cpu_segment(
        ggml_backend_xtc_context * ctx,
        struct ggml_cgraph * cgraph,
        int i0,
        int i1) {
    if (i0 >= i1) {
        return GGML_STATUS_SUCCESS;
    }

    struct ggml_cgraph segment = ggml_xtc_graph_view(cgraph, i0, i1);
    return ggml_backend_graph_compute(ctx->cpu, &segment);
}

static bool ggml_xtc_device_supports_op(ggml_backend_dev_t cpu_dev, const struct ggml_tensor * op) {
    if (ggml_xtc_is_disabled()) {
        return false;
    }

    ggml_xtc_matmul_shape shape;
    if (ggml_xtc_is_supported_matmul(op, &shape)) {
        if (!ggml_xtc_missing_falls_back_to_cpu()) {
            // claim every supported matmul: a missing kernel is a clean failure in graph_compute
            return true;
        }
        std::lock_guard<std::mutex> lock(g_xtc_mutex);
        if (ggml_xtc_get_matmul_kernel(shape).state == ggml_xtc_kernel_state::FOUND) {
            return true;
        }
    }

    return ggml_backend_dev_supports_op(cpu_dev, op);
}

// backend interface

static const char * ggml_backend_xtc_get_name(ggml_backend_t backend) {
    return "XTC";

    GGML_UNUSED(backend);
}

static void ggml_backend_xtc_free(ggml_backend_t backend) {
    ggml_backend_xtc_context * ctx = (ggml_backend_xtc_context *) backend->context;
    if (ctx->cpu) {
        ggml_backend_free(ctx->cpu);
    }
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_xtc_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_xtc_context * ctx = (ggml_backend_xtc_context *) backend->context;

    const bool to_cpu = ggml_xtc_missing_falls_back_to_cpu();

    // pass 1: resolve every kernel *before* computing anything, so that a missing kernel
    // fails the whole graph cleanly (and we can report all the missing ones at once)
    std::vector<ggml_xtc_kernel_fn>      fns(cgraph->n_nodes, nullptr);
    std::vector<ggml_xtc_missing_entry>  missing;
    std::vector<std::string>             broken;
    {
        std::lock_guard<std::mutex> lock(g_xtc_mutex);
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];
            if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_xtc_node_is_noop(node)) {
                continue;
            }

            ggml_xtc_matmul_shape shape;
            if (!ggml_xtc_is_supported_matmul(node, &shape)) {
                continue;
            }

            const ggml_xtc_kernel & k = ggml_xtc_get_matmul_kernel(shape);
            switch (k.state) {
                case ggml_xtc_kernel_state::FOUND:
                    fns[i] = k.fn;
                    break;
                case ggml_xtc_kernel_state::BROKEN:
                    if (to_cpu) {
                        GGML_LOG_WARN("%s: broken kernel %s (%s), using the CPU\n", __func__, k.key.c_str(), k.error.c_str());
                    } else {
                        broken.push_back(k.key + ": " + k.error);
                    }
                    break;
                case ggml_xtc_kernel_state::MISSING:
                    if (!to_cpu) {
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
                    }
                    break;
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

    // pass 2: XTC kernels for the resolved nodes, CPU for the rest
    int cpu_i0 = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (fns[i] == nullptr) {
            continue;
        }

        enum ggml_status status = ggml_xtc_flush_cpu_segment(ctx, cgraph, cpu_i0, i);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }

        ggml_xtc_run_mul_mat_kernel(fns[i], cgraph->nodes[i]);
        cpu_i0 = i + 1;
    }

    return ggml_xtc_flush_cpu_segment(ctx, cgraph, cpu_i0, cgraph->n_nodes);
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
    ctx->cpu = ggml_backend_cpu_init();
    if (ctx->cpu == nullptr) {
        delete ctx;
        return nullptr;
    }

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
    ggml_backend_cpu_set_n_threads(ctx->cpu, n_threads);
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
    ggml_backend_dev_t cpu_dev = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    return ggml_xtc_device_supports_op(cpu_dev, op);

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

static void * ggml_backend_xtc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) ggml_backend_xtc_set_n_threads;
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
