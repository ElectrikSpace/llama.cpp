#include "ggml-xtc-mppa.h"

#include "ggml-xtc-common.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// MPPA device backend. Tensors live in MPPA memory: each buffer is one device allocation of the MPPA runtime
// (xtc/src/xtc/csrcs/runtimes/accelerator/mppa/host.c), loaded from XTC_MPPA_RUNTIME.
// Compute: MUL_MAT and same-shape F32 ADD / MUL run as MPPA kernels on device memory. No other operator is
// claimed and there is no fallback: the scheduler runs them on another backend.

#define GGML_XTC_MPPA_NAME "XTC_MPPA"

// must match MPPA_MEMORY_KIND_GLOBAL in host.c
static constexpr int    GGML_XTC_MPPA_MEMORY_KIND_GLOBAL = 0;
static constexpr size_t GGML_XTC_MPPA_ALIGNMENT          = 64;
static constexpr size_t GGML_XTC_MPPA_CHUNK              = 1 << 20;

struct ggml_xtc_mppa_runtime {
    bool (*init_device)(void);
    void * (*create_memory_handle)(void);
    bool (*destroy_memory_handle)(void *);
    bool (*memory_allocate)(void *, size_t, int);
    bool (*memory_free)(void *, int);
    bool (*memory_copy_to_offset)(void *, void *, size_t, size_t);
    bool (*memory_copy_from_offset)(void *, void *, size_t, size_t);
    void * (*get_common_structures)(void);
    void * (*memory_data_pointer)(void *);
};

template <typename Fn>
static Fn ggml_xtc_mppa_sym(void * lib, const char * name) {
    void * sym = dlsym(lib, name);
    if (sym == nullptr) {
        GGML_ABORT("ggml-xtc-mppa: symbol %s not found: %s", name, dlerror());
    }
    return reinterpret_cast<Fn>(sym);
}

static const ggml_xtc_mppa_runtime & ggml_xtc_mppa_load_runtime() {
    static const ggml_xtc_mppa_runtime rt = [] {
        const char * path = getenv("XTC_MPPA_RUNTIME");
        if (path == nullptr || path[0] == '\0') {
            GGML_ABORT("ggml-xtc-mppa: XTC_MPPA_RUNTIME must point to the MPPA runtime (mppa_runtime_host.so)");
        }
        void * lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) {
            GGML_ABORT("ggml-xtc-mppa: cannot load %s: %s", path, dlerror());
        }
        ggml_xtc_mppa_runtime r = {};
        r.init_device             = ggml_xtc_mppa_sym<decltype(r.init_device)>(lib, "mppa_init_device");
        r.create_memory_handle    = ggml_xtc_mppa_sym<decltype(r.create_memory_handle)>(lib, "mppa_create_memory_handle");
        r.destroy_memory_handle   = ggml_xtc_mppa_sym<decltype(r.destroy_memory_handle)>(lib, "mppa_destroy_memory_handle");
        r.memory_allocate         = ggml_xtc_mppa_sym<decltype(r.memory_allocate)>(lib, "mppa_memory_allocate");
        r.memory_free             = ggml_xtc_mppa_sym<decltype(r.memory_free)>(lib, "mppa_memory_free");
        r.memory_copy_to_offset   = ggml_xtc_mppa_sym<decltype(r.memory_copy_to_offset)>(lib, "mppa_memory_copy_to_offset");
        r.memory_copy_from_offset = ggml_xtc_mppa_sym<decltype(r.memory_copy_from_offset)>(lib, "mppa_memory_copy_from_offset");
        r.get_common_structures   = ggml_xtc_mppa_sym<decltype(r.get_common_structures)>(lib, "mppa_get_common_structures");
        r.memory_data_pointer     = ggml_xtc_mppa_sym<decltype(r.memory_data_pointer)>(lib, "mppa_memory_data_pointer");
        if (!r.init_device()) {
            GGML_ABORT("ggml-xtc-mppa: MPPA device initialization failed");
        }
        return r;
    }();
    return rt;
}

struct ggml_xtc_mppa_buffer {
    void *    handle; // MPPA memory handle
    size_t    size;
    uintptr_t base;   // fake base address: tensor->data = base + offset, never dereferenced
};

// fake base addresses are handed out in disjoint ranges, separated by one chunk
static std::atomic<uintptr_t> g_xtc_mppa_next_base{ (uintptr_t) 1 << 40 };

static size_t ggml_xtc_mppa_device_offset(const ggml_xtc_mppa_buffer * b, const struct ggml_tensor * t,
                                          size_t offset, size_t size) {
    const uintptr_t addr = (uintptr_t) t->data + offset;
    GGML_ASSERT(addr >= b->base && addr - b->base + size <= b->size);
    return addr - b->base;
}

static void ggml_xtc_mppa_buffer_free(ggml_backend_buffer_t buffer) {
    const auto & rt = ggml_xtc_mppa_load_runtime();
    auto * b = (ggml_xtc_mppa_buffer *) buffer->context;
    rt.memory_free(b->handle, GGML_XTC_MPPA_MEMORY_KIND_GLOBAL);
    rt.destroy_memory_handle(b->handle);
    delete b;
}

static void * ggml_xtc_mppa_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *) ((ggml_xtc_mppa_buffer *) buffer->context)->base;
}

static void ggml_xtc_mppa_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                               uint8_t value, size_t offset, size_t size) {
    const auto & rt = ggml_xtc_mppa_load_runtime();
    auto * b = (ggml_xtc_mppa_buffer *) buffer->context;
    const size_t dev_offset = ggml_xtc_mppa_device_offset(b, tensor, offset, size);
    std::vector<uint8_t> chunk(std::min(size, GGML_XTC_MPPA_CHUNK), value);
    for (size_t done = 0; done < size; done += chunk.size()) {
        const size_t n = std::min(chunk.size(), size - done);
        GGML_ASSERT(rt.memory_copy_to_offset(b->handle, chunk.data(), dev_offset + done, n));
    }
}

static void ggml_xtc_mppa_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                            const void * data, size_t offset, size_t size) {
    const auto & rt = ggml_xtc_mppa_load_runtime();
    auto * b = (ggml_xtc_mppa_buffer *) buffer->context;
    const size_t dev_offset = ggml_xtc_mppa_device_offset(b, tensor, offset, size);
    GGML_ASSERT(rt.memory_copy_to_offset(b->handle, const_cast<void *>(data), dev_offset, size));
}

static void ggml_xtc_mppa_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                                            void * data, size_t offset, size_t size) {
    const auto & rt = ggml_xtc_mppa_load_runtime();
    auto * b = (ggml_xtc_mppa_buffer *) buffer->context;
    const size_t dev_offset = ggml_xtc_mppa_device_offset(b, tensor, offset, size);
    GGML_ASSERT(rt.memory_copy_from_offset(b->handle, data, dev_offset, size));
}

// the source is copied through the host: the MPPA runtime has no device to device copy yet
static bool ggml_xtc_mppa_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src,
                                            struct ggml_tensor * dst) {
    const bool src_is_mppa = ggml_backend_buffer_get_type(src->buffer)->iface.get_name ==
                             ggml_backend_buffer_get_type(buffer)->iface.get_name;
    if (!ggml_backend_buffer_is_host(src->buffer) && !src_is_mppa) {
        return false;
    }
    std::vector<uint8_t> tmp(ggml_nbytes(src));
    ggml_backend_tensor_get(src, tmp.data(), 0, tmp.size());
    ggml_backend_tensor_set(dst, tmp.data(), 0, tmp.size());
    return true;
}

static void ggml_xtc_mppa_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const auto & rt = ggml_xtc_mppa_load_runtime();
    auto * b = (ggml_xtc_mppa_buffer *) buffer->context;
    std::vector<uint8_t> chunk(std::min(b->size, GGML_XTC_MPPA_CHUNK), value);
    for (size_t done = 0; done < b->size; done += chunk.size()) {
        const size_t n = std::min(chunk.size(), b->size - done);
        GGML_ASSERT(rt.memory_copy_to_offset(b->handle, chunk.data(), done, n));
    }
}

static const ggml_backend_buffer_i ggml_xtc_mppa_buffer_i = {
    /* .free_buffer   = */ ggml_xtc_mppa_buffer_free,
    /* .get_base      = */ ggml_xtc_mppa_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_xtc_mppa_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_xtc_mppa_buffer_set_tensor,
    /* .get_tensor    = */ ggml_xtc_mppa_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ggml_xtc_mppa_buffer_cpy_tensor,
    /* .clear         = */ ggml_xtc_mppa_buffer_clear,
    /* .reset         = */ nullptr,
    /* .set_tensor_n  = */ nullptr,
    /* .get_tensor_n  = */ nullptr,
    /* .cpy_tensor_n  = */ nullptr,
};

static const char * ggml_xtc_mppa_buft_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_XTC_MPPA_NAME;
}

static ggml_backend_buffer_t ggml_xtc_mppa_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (size == 0) {
        return nullptr;
    }
    const auto & rt = ggml_xtc_mppa_load_runtime();
    void * handle = rt.create_memory_handle();
    if (!rt.memory_allocate(handle, size, GGML_XTC_MPPA_MEMORY_KIND_GLOBAL)) {
        rt.destroy_memory_handle(handle);
        GGML_ABORT("ggml-xtc-mppa: cannot allocate %zu bytes on the MPPA", size);
    }
    // each fake base starts on a chunk boundary
    const size_t span = (size + GGML_XTC_MPPA_CHUNK - 1) / GGML_XTC_MPPA_CHUNK * GGML_XTC_MPPA_CHUNK + GGML_XTC_MPPA_CHUNK;
    auto * b = new ggml_xtc_mppa_buffer{ handle, size, g_xtc_mppa_next_base.fetch_add(span) };
    return ggml_backend_buffer_init(buft, ggml_xtc_mppa_buffer_i, b, size);
}

static size_t ggml_xtc_mppa_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_XTC_MPPA_ALIGNMENT;
}

static bool ggml_xtc_mppa_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static const ggml_backend_buffer_type_i ggml_xtc_mppa_buft_i = {
    /* .get_name         = */ ggml_xtc_mppa_buft_get_name,
    /* .alloc_buffer     = */ ggml_xtc_mppa_buft_alloc_buffer,
    /* .alloc_buffer_n   = */ nullptr,
    /* .get_alignment    = */ ggml_xtc_mppa_buft_get_alignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ nullptr,
    /* .get_alloc_size_n = */ nullptr,
    /* .is_host          = */ ggml_xtc_mppa_buft_is_host,
};

static bool ggml_xtc_mppa_is_buffer(const ggml_backend_buffer_t buffer) {
    return buffer != nullptr && ggml_backend_buffer_get_type(buffer)->iface.get_name == ggml_xtc_mppa_buft_get_name;
}

// Device address of byte_off bytes from the data of an MPPA tensor. The kernels take the device address of
// each tensor: it is the voffset of the buffer, plus the offset of the tensor in the buffer.
static uintptr_t ggml_xtc_mppa_dev_addr(const struct ggml_tensor * t, size_t byte_off = 0) {
    GGML_ASSERT(ggml_xtc_mppa_is_buffer(t->buffer));
    auto * b = (ggml_xtc_mppa_buffer *) t->buffer->context;
    const uintptr_t addr = (uintptr_t) t->data + byte_off;
    GGML_ASSERT(addr >= b->base && addr - b->base < b->size);
    return (uintptr_t) ggml_xtc_mppa_load_runtime().memory_data_pointer(b->handle) + (addr - b->base);
}

// MPPA kernels: kernel.so of a kernel base entry built for the mppa target (KernelBase.add(target="mppa")).
// Like MppaDevice, at most GGML_XTC_MPPA_MAX_KERNELS are loaded at once.
static constexpr size_t GGML_XTC_MPPA_MAX_KERNELS = 10;

struct ggml_xtc_mppa_kernel {
    void *             lib      = nullptr;
    ggml_xtc_kernel_fn fn       = nullptr;
    uint64_t           last_use = 0;
};

// protects the kernel table and the graph generation: held by graph_compute
static std::mutex g_xtc_mppa_mutex;
static std::map<std::string, ggml_xtc_mppa_kernel> g_xtc_mppa_kernels; // loaded kernels, by key
static uint64_t g_xtc_mppa_generation = 0;                             // one per graph_compute

// Loads the kernel of a shape (g_xtc_mppa_mutex must be held). A kernel of the running graph is never evicted.
// Returns nullptr and sets err when there is no such kernel or no room for it.
static const ggml_xtc_mppa_kernel * ggml_xtc_mppa_get_kernel(const ggml_xtc_kernel_shape & shape, std::string & err) {
    const std::string key = ggml_xtc_kernel_key(shape, "mppa");
    auto it = g_xtc_mppa_kernels.find(key);
    if (it != g_xtc_mppa_kernels.end()) {
        it->second.last_use = g_xtc_mppa_generation;
        return &it->second;
    }

    const fs::path dir = ggml_xtc_kernelbase_root() / key;
    const fs::path so  = dir / "kernel.so";
    if (!fs::exists(so)) {
        err = key + " (" + ggml_xtc_shape_str(shape) + "): no MPPA kernel in " + dir.string();
        return nullptr;
    }

    if (g_xtc_mppa_kernels.size() >= GGML_XTC_MPPA_MAX_KERNELS) {
        auto victim = g_xtc_mppa_kernels.end();
        for (auto e = g_xtc_mppa_kernels.begin(); e != g_xtc_mppa_kernels.end(); ++e) {
            if (e->second.last_use < g_xtc_mppa_generation &&
                (victim == g_xtc_mppa_kernels.end() || e->second.last_use < victim->second.last_use)) {
                victim = e;
            }
        }
        if (victim == g_xtc_mppa_kernels.end()) {
            err = "more than " + std::to_string(GGML_XTC_MPPA_MAX_KERNELS) + " MPPA kernels needed by one graph";
            return nullptr;
        }
        dlclose(victim->second.lib);
        g_xtc_mppa_kernels.erase(victim);
    }

    std::string sym = ggml_xtc_meta_value(dir / "meta.yaml", "payload_name");
    if (sym.empty()) {
        sym = ggml_xtc_op_name(shape.kop);
    }

    void * lib = dlopen(so.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) {
        err = "cannot load " + so.string() + ": " + dlerror();
        return nullptr;
    }
    using set_common_fn = void (*)(void *);
    using load_fn       = void (*)(void);
    auto set_common = (set_common_fn) dlsym(lib, "set_mppa_common_structures");
    auto load       = (load_fn) dlsym(lib, "load_kernel");
    auto fn         = (ggml_xtc_kernel_fn) dlsym(lib, sym.c_str());
    if (set_common == nullptr || load == nullptr || fn == nullptr) {
        dlclose(lib);
        err = "symbol '" + sym + "' or MPPA loader of " + so.string() + " not found";
        return nullptr;
    }
    // same order as MppaDevice.load_module: the kernel shares the device context of the runtime
    set_common(ggml_xtc_mppa_load_runtime().get_common_structures());
    load();

    ggml_xtc_mppa_kernel k;
    k.lib      = lib;
    k.fn       = fn;
    k.last_use = g_xtc_mppa_generation;
    return &g_xtc_mppa_kernels.emplace(key, k).first->second;
}

// Kernel of a node, when the node is one kernel call on MPPA tensors (see ggml_xtc_is_supported_matmul and
// ggml_xtc_is_flat_eltwise): no fallback, other nodes are not claimed.
static bool ggml_xtc_mppa_node_shape(const struct ggml_tensor * op, ggml_xtc_kernel_shape * s) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return ggml_xtc_is_supported_matmul(op, s);
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            if (ggml_nelements(op) == 0 || !ggml_xtc_is_flat_eltwise(op)) {
                return false;
            }
            s->kop = op->op == GGML_OP_ADD ? XTC_KOP_ADD : XTC_KOP_MUL;
            s->m   = op->ne[1] * op->ne[2] * op->ne[3];
            s->n   = op->ne[0];
            return true;
        default:
            return false;
    }
}

// C[m,n] = A[m,k] x B[k,n] per slice, like ggml_xtc_run_mul_mat_kernel but on device addresses
static void ggml_xtc_mppa_run_matmul(const ggml_xtc_mppa_kernel & k, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;

            const uintptr_t a = ggml_xtc_mppa_dev_addr(src1, i12 * nb12 + i13 * nb13);
            const uintptr_t b = ggml_xtc_mppa_dev_addr(src0, i02 * nb02 + i03 * nb03);
            const uintptr_t c = ggml_xtc_mppa_dev_addr(dst, i12 * nb2 + i13 * nb3);
            k.fn((void *) a, (void *) b, (void *) c);
        }
    }
}

// flat ADD / MUL: one call on the whole (M, N) tensor
static void ggml_xtc_mppa_run_eltwise(const ggml_xtc_mppa_kernel & k, struct ggml_tensor * dst) {
    const uintptr_t a = ggml_xtc_mppa_dev_addr(dst->src[0]);
    const uintptr_t b = ggml_xtc_mppa_dev_addr(dst->src[1]);
    const uintptr_t d = ggml_xtc_mppa_dev_addr(dst);
    k.fn((void *) a, (void *) b, (void *) d);
}

static enum ggml_status ggml_backend_xtc_mppa_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);

    std::lock_guard<std::mutex> lock(g_xtc_mppa_mutex);
    g_xtc_mppa_generation++;

    // pass 1: resolve every kernel and check the tensors *before* computing anything
    std::vector<const ggml_xtc_mppa_kernel *> fns(cgraph->n_nodes, nullptr);
    std::vector<std::string> errors;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_op_is_empty(node->op)) {
            continue;
        }
        ggml_xtc_kernel_shape shape;
        if (!ggml_xtc_mppa_node_shape(node, &shape)) {
            errors.push_back(std::string(ggml_op_desc(node)) + " (" + node->name + ") is not supported by XTC_MPPA");
            continue;
        }
        bool on_device = ggml_xtc_mppa_is_buffer(node->buffer);
        for (const struct ggml_tensor * src : node->src) {
            on_device = on_device && (src == nullptr || ggml_xtc_mppa_is_buffer(src->buffer));
        }
        if (!on_device) {
            errors.push_back(std::string(node->name) + ": tensors of " + ggml_op_desc(node) + " are not in XTC_MPPA memory");
            continue;
        }
        std::string err;
        fns[i] = ggml_xtc_mppa_get_kernel(shape, err);
        if (fns[i] == nullptr) {
            errors.push_back(err);
        }
    }
    if (!errors.empty()) {
        for (const auto & e : errors) {
            GGML_LOG_ERROR("%s: %s\n", __func__, e.c_str());
        }
        return GGML_STATUS_FAILED;
    }

    // pass 2: the kernels of this graph are loaded and cannot be evicted
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (fns[i] == nullptr) {
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT) {
            ggml_xtc_mppa_run_matmul(*fns[i], node);
        } else {
            ggml_xtc_mppa_run_eltwise(*fns[i], node);
        }
    }

    return GGML_STATUS_SUCCESS;
}

static const char * ggml_backend_xtc_mppa_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_XTC_MPPA_NAME;
}

static void ggml_backend_xtc_mppa_free(ggml_backend_t backend) {
    delete backend;
}

static struct ggml_backend_i ggml_backend_xtc_mppa_i = {
    /* .get_name                = */ ggml_backend_xtc_mppa_get_name,
    /* .free                    = */ ggml_backend_xtc_mppa_free,
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
    /* .graph_compute           = */ ggml_backend_xtc_mppa_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_xtc_mppa_guid(void) {
    static ggml_guid guid = { 0x4d, 0x50, 0x50, 0x41, 0x7b, 0x12, 0x4e, 0x90, 0xa3, 0x5c, 0x2f, 0x81, 0x66, 0xd7, 0x08, 0x19 };
    return &guid;
}

static const char * ggml_xtc_mppa_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_XTC_MPPA_NAME;
}

static const char * ggml_xtc_mppa_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "XTC MPPA accelerator";
}

// TODO: device memory is not reported yet (the runtime has no query for it)
static void ggml_xtc_mppa_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_xtc_mppa_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_xtc_mppa_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_xtc_mppa_device_get_name(dev);
    props->description = ggml_xtc_mppa_device_get_description(dev);
    props->type        = ggml_xtc_mppa_device_get_type(dev);
    ggml_xtc_mppa_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = nullptr;
    props->caps      = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t ggml_xtc_mppa_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_xtc_mppa_load_runtime(); // device initialization
    return new ggml_backend{
        /* .guid    = */ ggml_backend_xtc_mppa_guid(),
        /* .iface   = */ ggml_backend_xtc_mppa_i,
        /* .device  = */ dev,
        /* .context = */ nullptr,
    };
}

static ggml_backend_buffer_type_t ggml_xtc_mppa_device_get_buffer_type(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type buft = {
        /* .iface   = */ ggml_xtc_mppa_buft_i,
        /* .device  = */ dev,
        /* .context = */ nullptr,
    };
    return &buft;
}

static bool ggml_xtc_mppa_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    // leaves (weights and inputs in MPPA memory) and views are not computed: the scheduler needs them claimed
    if (ggml_op_is_empty(op->op)) {
        return true;
    }
    ggml_xtc_kernel_shape shape;
    return ggml_xtc_mppa_node_shape(op, &shape);
}

static bool ggml_xtc_mppa_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return buft->iface.get_name == ggml_xtc_mppa_buft_get_name;
}

static const struct ggml_backend_device_i ggml_xtc_mppa_device_i = {
    /* .get_name             = */ ggml_xtc_mppa_device_get_name,
    /* .get_description      = */ ggml_xtc_mppa_device_get_description,
    /* .get_memory           = */ ggml_xtc_mppa_device_get_memory,
    /* .get_type             = */ ggml_xtc_mppa_device_get_type,
    /* .get_props            = */ ggml_xtc_mppa_device_get_props,
    /* .init_backend         = */ ggml_xtc_mppa_device_init_backend,
    /* .get_buffer_type      = */ ggml_xtc_mppa_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_xtc_mppa_device_supports_op,
    /* .supports_buft        = */ ggml_xtc_mppa_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static const char * ggml_xtc_mppa_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_XTC_MPPA_NAME;
}

static size_t ggml_xtc_mppa_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_xtc_mppa_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device dev = {
        /* .iface   = */ ggml_xtc_mppa_device_i,
        /* .reg     = */ nullptr,
        /* .context = */ nullptr,
    };
    dev.reg = reg;
    return &dev;
}

static const struct ggml_backend_reg_i ggml_xtc_mppa_reg_i = {
    /* .get_name         = */ ggml_xtc_mppa_reg_get_name,
    /* .get_device_count = */ ggml_xtc_mppa_reg_get_device_count,
    /* .get_device       = */ ggml_xtc_mppa_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_xtc_mppa_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_xtc_mppa_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}
