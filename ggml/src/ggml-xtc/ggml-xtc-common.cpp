#include "ggml-xtc-common.h"

#include "ggml-impl.h"
#include "xtc_support_lib.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

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

const char * ggml_xtc_op_name(ggml_xtc_kop kop) {
    switch (kop) {
        case XTC_KOP_ADD: return "add";
        case XTC_KOP_MUL: return "mul";
        case XTC_KOP_FMA: return "fma";
        case XTC_KOP_RSQRT: return "rsqrt";
        case XTC_KOP_SILU: return "silu";
        case XTC_KOP_CAST_F16: return "cast";
        case XTC_KOP_EXP: return "exp";
        case XTC_KOP_REDUCE_MAX: return "reduce_max";
        case XTC_KOP_REDUCE_SUM: return "reduce_sum";
        case XTC_KOP_SUB_ROW: return "sub_row";
        case XTC_KOP_DIV_ROW: return "div_row";
        default:          return "matmul";
    }
}

bool ggml_xtc_kop_is_matmul(ggml_xtc_kop kop) {
    return kop == XTC_KOP_MATMUL || kop == XTC_KOP_MATMUL_F16_BT || kop == XTC_KOP_MATMUL_F16;
}

ggml_xtc_kernel_desc ggml_xtc_describe(const ggml_xtc_kernel_shape & s) {
    ggml_xtc_kernel_desc d;
    const char * dt = s.f16 ? "float16" : "float32"; // dtype of the data
    switch (s.kop) {
        case XTC_KOP_MATMUL:
            d.inputs = { { dt, s.m, s.k, false }, { dt, s.k, s.n, true } };
            d.attr_dtype = s.f16 ? "float16" : nullptr;
            break;
        case XTC_KOP_MATMUL_F16_BT:
            d.inputs = { { "float16", s.m, s.k, false }, { "float16", s.k, s.n, true } };
            d.attr_dtype = dt;
            break;
        case XTC_KOP_MATMUL_F16:
            d.inputs = { { "float16", s.m, s.k, false }, { "float16", s.k, s.n, false } };
            d.attr_dtype = dt;
            break;
        case XTC_KOP_ADD:
        case XTC_KOP_MUL:
            d.inputs = { { dt, s.m, s.n, false }, { dt, s.m, s.n, false } };
            break;
        case XTC_KOP_RSQRT:
        case XTC_KOP_SILU:
        case XTC_KOP_EXP:
        case XTC_KOP_REDUCE_MAX:
        case XTC_KOP_REDUCE_SUM:
            d.inputs = { { dt, s.m, s.n, false } };
            break;
        case XTC_KOP_CAST_F16:
            d.inputs = { { dt, s.m, s.n, false } };
            d.attr_dtype = "float16";
            break;
        case XTC_KOP_SUB_ROW:
        case XTC_KOP_DIV_ROW:
            d.inputs = { { dt, s.m, s.n, false }, { dt, s.m, 0, false } };
            break;
        case XTC_KOP_FMA:
            d.inputs = { { dt, s.m, s.n, false }, { dt, s.m, s.n, false }, { dt, s.m, s.n, false } };
            break;
    }
    return d;
}

std::string ggml_xtc_shape_str(const ggml_xtc_kernel_shape & s) {
    char buf[128];
    if (ggml_xtc_kop_is_matmul(s.kop)) {
        snprintf(buf, sizeof(buf), "%s%s M=%" PRId64 " N=%" PRId64 " K=%" PRId64, ggml_xtc_op_name(s.kop),
                 s.f16 ? " f16" : s.kop == XTC_KOP_MATMUL ? " f32" : " f16->f32", s.m, s.n, s.k);
    } else {
        snprintf(buf, sizeof(buf), "%s %s%s M=%" PRId64 " N=%" PRId64, ggml_xtc_op_name(s.kop), s.f16 ? "f16" : "f32",
                 s.kop == XTC_KOP_CAST_F16 && !s.f16 ? "->f16" : "", s.m, s.n);
    }
    return buf;
}

// keys sorted, no spaces: json.dumps(sort_keys=True, separators=(",", ":")); target empty for host kernels
static std::string ggml_xtc_canonical_json(const ggml_xtc_kernel_shape & s, const std::string & target) {
    const ggml_xtc_kernel_desc d = ggml_xtc_describe(s);
    const int n_in = (int) d.inputs.size();
    std::string json = "{\"inputs\":[";
    for (int i = 0; i < n_in; i++) {
        const ggml_xtc_input_desc & in = d.inputs[i];
        char buf[256];
        const std::string dims = in.d1 > 0 ? std::to_string(in.d0) + "," + std::to_string(in.d1) : std::to_string(in.d0);
        snprintf(buf, sizeof(buf), "%s{\"expr\":{\"type\":{\"dtype\":\"%s\",%s\"shape\":[%s]}},\"uid\":\"%%%d\"}",
                 i ? "," : "", in.dtype, in.transposed ? "\"layout\":[1,0]," : "", dims.c_str(), i);
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
    if (!target.empty()) {
        json.pop_back();
        json += ",\"target\":\"" + target + "\"}";
    }
    return json;
}

std::string ggml_xtc_kernel_key(const ggml_xtc_kernel_shape & s, const std::string & target) {
    return ggml_xtc_sha256_hex(ggml_xtc_canonical_json(s, target)).substr(0, 16);
}

// graph.yaml for the stub (json is valid yaml; same dict XTCGraph.to_dict() produces)
std::string ggml_xtc_graph_yaml(const ggml_xtc_kernel_shape & s) {
    const ggml_xtc_kernel_desc d = ggml_xtc_describe(s);
    const std::string op = ggml_xtc_op_name(s.kop);
    std::string yaml = "name: " + op + "\nops_version: unknown\ninputs:\n";
    std::string args;
    for (size_t i = 0; i < d.inputs.size(); i++) {
        const ggml_xtc_input_desc & in = d.inputs[i];
        const std::string uid = op + "_in" + std::to_string(i);
        char buf[256];
        const std::string dims = in.d1 > 0 ? std::to_string(in.d0) + ", " + std::to_string(in.d1) : std::to_string(in.d0);
        snprintf(buf, sizeof(buf), "- uid: %s\n  name: %c\n  expr:\n    type: {shape: [%s], dtype: %s%s}\n",
                 uid.c_str(), (char) ('A' + i), dims.c_str(), in.dtype, in.transposed ? ", layout: [1, 0]" : "");
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

std::mutex g_xtc_mutex;
static std::map<ggml_xtc_kernel_shape, ggml_xtc_kernel> g_xtc_kernels;
static std::map<std::string, const xtc_kernel_entry *> g_xtc_bundle;
static bool                                           g_xtc_bundle_checked = false;

fs::path ggml_xtc_kernelbase_root() {
    const char * env = getenv("XTC_KERNELBASE");
    fs::path root = (env != nullptr && env[0] != '\0') ? fs::path(env) : fs::path("xtc_kernelbase");
    std::error_code ec;
    fs::path abs = fs::absolute(root, ec);
    return ec ? root : abs;
}

bool ggml_xtc_bundle_only() {
    const char * env = getenv("GGML_XTC_BUNDLE_ONLY");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0;
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
std::string ggml_xtc_meta_value(const fs::path & meta, const std::string & key) {
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
static void ggml_xtc_load_bundle() {
    if (g_xtc_bundle_checked) {
        return;
    }
    g_xtc_bundle_checked = true;

    const char * env = getenv("XTC_KERNEL_LIB");
    const fs::path path = (env != nullptr && env[0] != '\0') ? fs::path(env) : ggml_xtc_kernelbase_root() / "libxtc_kernels.so";
    std::error_code ec;
    if (!fs::exists(path, ec)) {
        return;
    }
    void * handle = ggml_xtc_dlopen(path.string().c_str());
    if (handle == nullptr) {
        GGML_LOG_WARN("%s: cannot load %s: %s\n", __func__, path.string().c_str(), ggml_xtc_dlerror().c_str());
        return;
    }
    const uint32_t *         version = (const uint32_t *) ggml_xtc_dlsym(handle, "xtc_kernel_abi_version");
    const size_t *           n       = (const size_t *) ggml_xtc_dlsym(handle, "xtc_kernel_registry_size");
    const xtc_kernel_entry * reg     = (const xtc_kernel_entry *) ggml_xtc_dlsym(handle, "xtc_kernel_registry");
    if (version == nullptr || n == nullptr || reg == nullptr || *version != XTC_KERNEL_ABI_VERSION) {
        GGML_LOG_WARN("%s: %s is not a kernel bundle of ABI version %d\n", __func__, path.string().c_str(),
                      XTC_KERNEL_ABI_VERSION);
        return;
    }
    for (size_t i = 0; i < *n; i++) {
        g_xtc_bundle[reg[i].key] = &reg[i];
    }
    GGML_LOG_INFO("%s: %zu kernel(s) from %s\n", __func__, *n, path.string().c_str());
}

// g_xtc_mutex must be held
ggml_xtc_kernel & ggml_xtc_get_kernel(const ggml_xtc_kernel_shape & shape) {
    auto it = g_xtc_kernels.find(shape);
    if (it != g_xtc_kernels.end()) {
        return it->second;
    }

    ggml_xtc_kernel k;
    k.key      = ggml_xtc_kernel_key(shape);
    k.n_inputs = (int) ggml_xtc_describe(shape).inputs.size();

    ggml_xtc_load_bundle();
    const auto entry = g_xtc_bundle.find(k.key);
    if (entry != g_xtc_bundle.end()) {
        if (entry->second->n_inputs != k.n_inputs) {
            k.state = ggml_xtc_kernel_state::BROKEN;
            k.error = "bundle entry has " + std::to_string(entry->second->n_inputs) + " inputs, expected " +
                      std::to_string(k.n_inputs);
        } else {
            k.state = ggml_xtc_kernel_state::FOUND;
            k.call  = entry->second->call;
        }
        return g_xtc_kernels.emplace(shape, std::move(k)).first->second;
    }

    const fs::path dir  = ggml_xtc_kernelbase_root() / k.key;
    const fs::path meta = dir / "meta.yaml";
    const fs::path so   = dir / "kernel.so";

    if (!ggml_xtc_bundle_only() && fs::exists(meta) && fs::exists(so)) {
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

bool ggml_xtc_is_supported_matmul(const struct ggml_tensor * op, ggml_xtc_kernel_shape * shape) {
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
    // (llama.cpp probes a weight buffer type with a 0-byte dummy buffer of usage ANY, before the weights are loaded)
    if (src0->buffer == nullptr) {
        return false;
    }
    const bool is_probe = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_ANY &&
                          ggml_backend_buffer_get_size(src0->buffer) == 0;
    if (ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS && !is_probe) {
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

bool ggml_xtc_is_flat_eltwise(const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    return src0 != nullptr && src1 != nullptr && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(op) &&
           ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, op);
}

// all the rows of the node and of its sources are packed, so one kernel call can take them all
bool ggml_xtc_flat_rows(const struct ggml_tensor * op) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = op->src[i];
        if (src != nullptr && (src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16) && src->ne[0] > 1 && !ggml_is_contiguous(src)) {
            return false;
        }
    }
    return ggml_is_contiguous(op);
}

std::vector<ggml_xtc_kernel_shape> ggml_xtc_node_kernels(const struct ggml_tensor * op) {
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

bool ggml_xtc_f32_rows(const struct ggml_tensor * t) {
    return t != nullptr && t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float);
}

bool ggml_xtc_f16_rows(const struct ggml_tensor * t) {
    return t != nullptr && t->type == GGML_TYPE_F16 && t->nb[0] == sizeof(uint16_t);
}

// packed rows of the activation type: F32 for the XTC backends, F16 for the F16 nodes of XTC_MPPA
static bool ggml_xtc_act_rows(const struct ggml_tensor * t, ggml_type type) {
    return type == GGML_TYPE_F16 ? ggml_xtc_f16_rows(t) : ggml_xtc_f32_rows(t);
}

// act: type of the activations (F32 or F16). F16: the attention output stays F32 (ggml definition) and SET_ROWS only
// writes an F16 cache.
static bool ggml_xtc_lib_supports_op_t(const struct ggml_tensor * op, ggml_type act) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const struct ggml_tensor * src2 = op->src[2];

    if (ggml_nelements(op) == 0) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_RMS_NORM:
            return ggml_xtc_act_rows(src0, act) && ggml_xtc_act_rows(op, act) && ggml_are_same_shape(src0, op);

        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return ggml_xtc_act_rows(src0, act) && ggml_xtc_act_rows(src1, act) && ggml_xtc_act_rows(op, act) &&
                   ggml_are_same_shape(src0, op) && ggml_can_repeat(src1, src0);

        case GGML_OP_GLU:
            // split SWIGLU only (gate = src0, up = src1)
            return ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && ggml_get_op_params_i32(op, 1) == 0 &&
                   ggml_xtc_act_rows(src0, act) && ggml_xtc_act_rows(src1, act) && ggml_xtc_act_rows(op, act) &&
                   ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, op);

        case GGML_OP_ROPE: {
            const int   n_dims = ggml_get_op_params_i32(op, 1);
            const int   mode   = ggml_get_op_params_i32(op, 2);
            float ext_factor;
            memcpy(&ext_factor, (const int32_t *) op->op_params + 7, sizeof(float));
            return ggml_xtc_act_rows(src0, act) && ggml_xtc_act_rows(op, act) && ggml_are_same_shape(src0, op) &&
                   src1 != nullptr && src1->type == GGML_TYPE_I32 && src1->ne[0] == src0->ne[2] &&
                   (mode == GGML_ROPE_TYPE_NORMAL || mode == GGML_ROPE_TYPE_NEOX) &&
                   ext_factor == 0.0f && ggml_get_op_params_i32(op, 15) == 0 &&
                   n_dims > 0 && n_dims % 2 == 0 && n_dims <= src0->ne[0] && n_dims <= XTC_MAX_HEAD_DIM &&
                   (src2 == nullptr || (src2->type == GGML_TYPE_F32 && src2->ne[0] >= n_dims / 2));
        }

        case GGML_OP_SET_ROWS:
            return ggml_xtc_act_rows(src0, act) && src1 != nullptr && src1->type == GGML_TYPE_I64 &&
                   (op->type == GGML_TYPE_F16 || (act == GGML_TYPE_F32 && op->type == GGML_TYPE_F32)) &&
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
            return ggml_xtc_act_rows(src0, act) && ggml_xtc_f32_rows(op) &&
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

bool ggml_xtc_lib_supports_op(const struct ggml_tensor * op) {
    return ggml_xtc_lib_supports_op_t(op, GGML_TYPE_F32);
}

bool ggml_xtc_is_flat_eltwise_f16(const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    return src0 != nullptr && src1 != nullptr && src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 &&
           op->type == GGML_TYPE_F16 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(op) &&
           ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, op);
}

std::vector<ggml_xtc_kernel_shape> ggml_xtc_node_kernels_f16(const struct ggml_tensor * op) {
    std::vector<ggml_xtc_kernel_shape> out;
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    if (src0 == nullptr) {
        return out;
    }
    auto k = [](ggml_xtc_kop kop, int64_t m, int64_t n, int64_t kk = 0, bool f16 = true) {
        ggml_xtc_kernel_shape s;
        s.kop = kop; s.m = m; s.n = n; s.k = kk; s.f16 = f16;
        return s;
    };

    switch (op->op) {
        case GGML_OP_MUL_MAT:
            // F16 weight x F16 input -> F32 output: the F16 x F16 -> F32 kernel, B (the weight) transposed
            if (src0->type == GGML_TYPE_F16 && src1 != nullptr && src1->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 && src0->buffer != nullptr &&
                ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(op) &&
                ggml_nelements(op) > 0 && ggml_nelements(src0) > 0 && ggml_nelements(src1) > 0 &&
                src1->ne[2] % src0->ne[2] == 0 && src1->ne[3] % src0->ne[3] == 0) {
                out.push_back(k(XTC_KOP_MATMUL_F16_BT, src1->ne[1], src0->ne[1], src0->ne[0], false));
            }
            break;
        case GGML_OP_ADD:
        case GGML_OP_MUL: {
            const ggml_xtc_kop kop = op->op == GGML_OP_ADD ? XTC_KOP_ADD : XTC_KOP_MUL;
            if (ggml_nelements(op) > 0 && ggml_xtc_is_flat_eltwise_f16(op)) {
                out.push_back(k(kop, op->ne[1] * op->ne[2] * op->ne[3], op->ne[0]));
            } else if (ggml_xtc_lib_supports_op_t(op, GGML_TYPE_F16) && src1->ne[0] == op->ne[0]) {
                out.push_back(k(kop, 1, op->ne[0]));
            }
            break;
        }
        case GGML_OP_ROPE:
            if (ggml_xtc_lib_supports_op_t(op, GGML_TYPE_F16) && src0->nb[1] == src0->ne[0] * sizeof(uint16_t) &&
                op->nb[1] == op->ne[0] * sizeof(uint16_t)) {
                const int64_t H = op->ne[1], D = op->ne[0];
                out.push_back(k(XTC_KOP_MATMUL, H, D, D));
                out.push_back(k(XTC_KOP_MUL, H, D));
                out.push_back(k(XTC_KOP_FMA, H, D));
            }
            break;
        case GGML_OP_RMS_NORM:
            if (ggml_xtc_lib_supports_op_t(op, GGML_TYPE_F16)) {
                const int64_t M = ggml_xtc_flat_rows(op) ? ggml_nrows(op) : 1;
                const int64_t N = op->ne[0];
                out.push_back(k(XTC_KOP_MUL, M, N));
                out.push_back(k(XTC_KOP_MATMUL, M, 1, N));
                out.push_back(k(XTC_KOP_ADD, M, 1));
                out.push_back(k(XTC_KOP_RSQRT, M, 1));
                out.push_back(k(XTC_KOP_MATMUL, M, N, 1));
                out.push_back(k(XTC_KOP_MUL, M, N));
            }
            break;
        case GGML_OP_GLU:
            if (ggml_xtc_lib_supports_op_t(op, GGML_TYPE_F16)) {
                const int64_t M = ggml_xtc_flat_rows(op) ? ggml_nrows(op) : 1;
                out.push_back(k(XTC_KOP_SILU, M, op->ne[0]));
                out.push_back(k(XTC_KOP_MUL, M, op->ne[0]));
            }
            break;
        case GGML_OP_SET_ROWS:
            if (ggml_xtc_lib_supports_op_t(op, GGML_TYPE_F16)) {
                out.push_back(k(XTC_KOP_CAST_F16, 1, op->ne[0]));
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            if (ggml_xtc_lib_supports_op_t(op, GGML_TYPE_F16) && src1->ne[1] > 0) {
                const int64_t M = (src0->ne[2] / src1->ne[2]) * src0->ne[1];
                const int64_t D = src0->ne[0];
                const int64_t N = src1->ne[1];
                out.push_back(k(XTC_KOP_MATMUL_F16_BT, M, N, D));
                out.push_back(k(XTC_KOP_FMA, M, N));
                out.push_back(k(XTC_KOP_REDUCE_MAX, M, N));
                out.push_back(k(XTC_KOP_SUB_ROW, M, N));
                out.push_back(k(XTC_KOP_EXP, M, N));
                out.push_back(k(XTC_KOP_REDUCE_SUM, M, N));
                out.push_back(k(XTC_KOP_DIV_ROW, M, N));
                out.push_back(k(XTC_KOP_MATMUL_F16, M, D, N));
            }
            break;
        case GGML_OP_CPY:
            // F32 -> F16 cast of a packed tensor: the cast after a F32 output (MUL_MAT, FLASH_ATTN_EXT)
            if (src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F16 && ggml_nelements(op) > 0 &&
                ggml_is_contiguous(src0) && ggml_is_contiguous(op) && ggml_are_same_shape(src0, op)) {
                out.push_back(k(XTC_KOP_CAST_F16, ggml_nrows(op), op->ne[0], 0, false));
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

void ggml_xtc_write_stub(const fs::path & root, ggml_xtc_missing_entry & e) {
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

std::string ggml_xtc_failure_message(const fs::path & root,
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
            "  3. xtc-kernelbase bundle --root '%s'  (or keep the per-entry kernel.so: found without it)\n"
            "  4. rerun llama.cpp with the same model and arguments\n"
            "Tips: kernels have a static shape and M is the number of tokens in the micro-batch, so each\n"
            "      micro-batch size needs its own kernels. Run with `-ub 1` to only need the M=1 kernels.\n"
            "      GGML_XTC_MISSING_KERNEL=cpu leaves those matmuls to the CPU backend instead of failing.\n",
            root.string().c_str(), root.string().c_str());
        msg += line;
    }
    for (const auto & b : broken) {
        msg += "ggml-xtc: broken kernel in the kernel base: " + b + "\n";
    }
    return msg;
}
