#include "ggml-xtc-mppa.h"

#include "ggml-xtc-common.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "xtc_support_lib.h"

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// MPPA device backend. Tensors live in MPPA memory: each buffer is one device allocation of the MPPA runtime
// (xtc/src/xtc/csrcs/runtimes/accelerator/mppa/host.c), loaded from XTC_MPPA_RUNTIME.
// Compute: the nodes that ggml_xtc_node_kernels splits into kernels run on device memory, one runner per op
// (see ggml_xtc_mppa_has_runner): MUL_MAT, ADD/MUL (flat, row broadcast, strided rows), RMS_NORM, SWIGLU,
// ROPE, F16 SET_ROWS and FLASH_ATTN_EXT, and the F16 flavours of them (F16 activations and weights, see
// ggml_xtc_node_kernels_f16) with the F32 -> F16 CPY after the F32 outputs of MUL_MAT and FLASH_ATTN_EXT. Glue work (constants, index reads, gathers, fix-ups) is staged through
// the host into a per-backend device scratch area. Other operators are not claimed and there is no fallback:
// the scheduler runs them on another backend.

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

// Host copies of the data of an MPPA tensor, at byte_off bytes from its data
static void ggml_xtc_mppa_tensor_read(const struct ggml_tensor * t, size_t byte_off, void * host,
                                                      size_t bytes) {
    GGML_ASSERT(ggml_xtc_mppa_is_buffer(t->buffer));
    const auto & rt = ggml_xtc_mppa_load_runtime();
    auto * b = (ggml_xtc_mppa_buffer *) t->buffer->context;
    const size_t dev_offset = ggml_xtc_mppa_device_offset(b, t, byte_off, bytes);
    GGML_ASSERT(rt.memory_copy_from_offset(b->handle, host, dev_offset, bytes));
}

static void ggml_xtc_mppa_tensor_write(struct ggml_tensor * t, size_t byte_off, const void * host,
                                                       size_t bytes) {
    GGML_ASSERT(ggml_xtc_mppa_is_buffer(t->buffer));
    const auto & rt = ggml_xtc_mppa_load_runtime();
    auto * b = (ggml_xtc_mppa_buffer *) t->buffer->context;
    const size_t dev_offset = ggml_xtc_mppa_device_offset(b, t, byte_off, bytes);
    GGML_ASSERT(rt.memory_copy_to_offset(b->handle, const_cast<void *>(host), dev_offset, bytes));
}

static size_t ggml_xtc_mppa_align(size_t bytes) {
    return (bytes + GGML_XTC_MPPA_ALIGNMENT - 1) / GGML_XTC_MPPA_ALIGNMENT * GGML_XTC_MPPA_ALIGNMENT;
}

// Device scratch of a backend for the temporaries of a node: one MPPA allocation, bump-allocated.
// reset() runs at the start of each graph_compute. reserve() may reallocate, so it must only run when no
// scratch address is in use: call it once per node, before alloc / upload.
struct ggml_xtc_mppa_context {
    void *    handle   = nullptr;
    size_t    capacity = 0;
    size_t    offset   = 0;
    uintptr_t base     = 0; // device address of the allocation

    ~ggml_xtc_mppa_context() {
        release();
    }

    void release() {
        if (handle != nullptr) {
            const auto & rt = ggml_xtc_mppa_load_runtime();
            rt.memory_free(handle, GGML_XTC_MPPA_MEMORY_KIND_GLOBAL);
            rt.destroy_memory_handle(handle);
        }
        handle   = nullptr;
        capacity = 0;
        offset   = 0;
        base     = 0;
    }

    void reset() {
        offset = 0;
    }

    // grows the scratch to at least total bytes; a growth also resets the bump offset
    void reserve(size_t total) {
        if (total <= capacity) {
            return;
        }
        release();
        const auto & rt = ggml_xtc_mppa_load_runtime();
        total  = ggml_xtc_mppa_align(total);
        handle = rt.create_memory_handle();
        if (!rt.memory_allocate(handle, total, GGML_XTC_MPPA_MEMORY_KIND_GLOBAL)) {
            rt.destroy_memory_handle(handle);
            handle = nullptr;
            GGML_ABORT("ggml-xtc-mppa: cannot allocate %zu bytes of scratch on the MPPA", total);
        }
        capacity = total;
        base     = (uintptr_t) rt.memory_data_pointer(handle);
    }

    // device address of a new block of bytes (the scratch must have room for it)
    uintptr_t alloc(size_t bytes) {
        GGML_ASSERT(offset + ggml_xtc_mppa_align(bytes) <= capacity);
        const uintptr_t addr = base + offset;
        offset += ggml_xtc_mppa_align(bytes);
        return addr;
    }

    uintptr_t upload(const void * host, size_t bytes) {
        const uintptr_t dev = alloc(bytes);
        write(dev, host, bytes);
        return dev;
    }

    // host to device at an existing scratch address
    void write(uintptr_t dev, const void * host, size_t bytes) {
        const auto & rt = ggml_xtc_mppa_load_runtime();
        GGML_ASSERT(rt.memory_copy_to_offset(handle, const_cast<void *>(host), dev_offset(dev, bytes), bytes));
    }

    // device to host from an existing scratch address
    void download(uintptr_t dev, void * host, size_t bytes) const {
        const auto & rt = ggml_xtc_mppa_load_runtime();
        GGML_ASSERT(rt.memory_copy_from_offset(handle, host, dev_offset(dev, bytes), bytes));
    }

private:
    size_t dev_offset(uintptr_t dev, size_t bytes) const {
        GGML_ASSERT(dev >= base && dev - base + bytes <= capacity);
        return dev - base;
    }
};

// MPPA kernels: kernel.so of a kernel base entry built for the mppa target (KernelBase.add(target="mppa")).
// Like MppaDevice, at most GGML_XTC_MPPA_MAX_KERNELS are loaded at once.
static constexpr size_t GGML_XTC_MPPA_MAX_KERNELS = 32;

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
        // free the device side of the kernel (its cluster memory) before closing the host library
        using unload_fn = void (*)(void);
        if (auto unload = (unload_fn) dlsym(victim->second.lib, "unload_kernel")) {
            unload();
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

// Ops with a runner below. Set an op to true here once its runner exists in ggml_backend_xtc_mppa_graph_compute.
static bool ggml_xtc_mppa_has_runner(const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_RMS_NORM:
        case GGML_OP_GLU:
        case GGML_OP_ROPE:
        case GGML_OP_SET_ROWS:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_CPY:
            return true;
        default:
            return false;
    }
}

// Kernels of a node: those of its F16 flavour, else the F32 ones
static std::vector<ggml_xtc_kernel_shape> ggml_xtc_mppa_node_kernels(const struct ggml_tensor * op) {
    std::vector<ggml_xtc_kernel_shape> shapes = ggml_xtc_node_kernels_f16(op);
    return shapes.empty() ? ggml_xtc_node_kernels(op) : shapes;
}

// F16 flavour of a node (F16 activations): its kernels and runners work on 2-byte elements
static bool ggml_xtc_mppa_is_f16(const struct ggml_tensor * op) {
    return !ggml_xtc_node_kernels_f16(op).empty();
}

// A node is claimed when it is a view or a leaf, or when it runs as kernels with a runner
static bool ggml_xtc_mppa_supports(const struct ggml_tensor * op) {
    if (ggml_op_is_empty(op->op)) {
        return true;
    }
    return ggml_xtc_mppa_has_runner(op) && !ggml_xtc_mppa_node_kernels(op).empty();
}

// Calls a kernel on device addresses: inputs first, then the output (2 to 4 pointers, like ggml_xtc_invoke)
static void ggml_xtc_mppa_invoke(const ggml_xtc_mppa_kernel & k, std::initializer_list<uintptr_t> args) {
    GGML_ASSERT(args.size() >= 2 && args.size() <= 4);
    void * p[4] = {};
    size_t n    = 0;
    for (const uintptr_t a : args) {
        p[n++] = (void *) a;
    }
    switch (n) {
        case 2: ((void (*)(void *, void *)) k.fn)(p[0], p[1]); break;
        case 3: k.fn(p[0], p[1], p[2]); break;
        case 4: ((void (*)(void *, void *, void *, void *)) k.fn)(p[0], p[1], p[2], p[3]); break;
        default: GGML_ABORT("%s: unexpected arity %zu", __func__, n);
    }
}

// C[m,n] = A[m,k] x B[k,n] per slice, like ggml_xtc_run_mul_mat_kernel but on device addresses
static void ggml_xtc_mppa_run_matmul(ggml_xtc_mppa_context & ctx, const std::vector<const ggml_xtc_mppa_kernel *> & fns,
                                     struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    const ggml_xtc_mppa_kernel & k = *fns[0];
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
            ggml_xtc_mppa_invoke(k, { a, b, c });
        }
    }
}

// ADD / MUL. Flat: one call on the whole (M, N) tensor. Else: one (1, N) call per row of dst, with src1 read
// at the broadcast row (i1 % ne11, i2 % ne12, i3 % ne13), like ggml_xtc_run_eltwise_kernel.
static void ggml_xtc_mppa_run_add_mul(ggml_xtc_mppa_context & ctx, const std::vector<const ggml_xtc_mppa_kernel *> & fns,
                                      struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    const ggml_xtc_mppa_kernel & k = *fns[0];
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (ggml_xtc_is_flat_eltwise(dst) || ggml_xtc_is_flat_eltwise_f16(dst)) {
        const uintptr_t a = ggml_xtc_mppa_dev_addr(src0);
        const uintptr_t b = ggml_xtc_mppa_dev_addr(src1);
        const uintptr_t d = ggml_xtc_mppa_dev_addr(dst);
        ggml_xtc_mppa_invoke(k, { a, b, d });
        return;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    for (int64_t i3 = 0; i3 < ne3; i3++) {
        for (int64_t i2 = 0; i2 < ne2; i2++) {
            for (int64_t i1 = 0; i1 < ne1; i1++) {
                const uintptr_t a = ggml_xtc_mppa_dev_addr(src0, i1 * nb01 + i2 * nb02 + i3 * nb03);
                const uintptr_t b = ggml_xtc_mppa_dev_addr(src1, (i1 % ne11) * nb11 + (i2 % ne12) * nb12 + (i3 % ne13) * nb13);
                const uintptr_t d = ggml_xtc_mppa_dev_addr(dst, i1 * nb1 + i2 * nb2 + i3 * nb3);
                ggml_xtc_mppa_invoke(k, { a, b, d });
            }
        }
    }
}

// byte offset of row (i1, i2, i3) of t
static size_t ggml_xtc_mppa_row_off(const struct ggml_tensor * t, int64_t i1, int64_t i2, int64_t i3) {
    return i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
}

// element size of the activations of a node: 2 for the F16 flavour, else 4
static size_t ggml_xtc_mppa_elem_size(bool f16) {
    return f16 ? sizeof(ggml_fp16_t) : sizeof(float);
}

// host floats written to the device as F16 or F32 elements, at a scratch address
static void ggml_xtc_mppa_write_elems(ggml_xtc_mppa_context & ctx, uintptr_t dev, const std::vector<float> & v, bool f16) {
    if (f16) {
        std::vector<ggml_fp16_t> h(v.size());
        ggml_fp32_to_fp16_row(v.data(), h.data(), (int64_t) v.size());
        ctx.write(dev, h.data(), h.size() * sizeof(ggml_fp16_t));
    } else {
        ctx.write(dev, v.data(), v.size() * sizeof(float));
    }
}

static uintptr_t ggml_xtc_mppa_upload_elems(ggml_xtc_mppa_context & ctx, const std::vector<float> & v, bool f16) {
    const uintptr_t dev = ctx.alloc(v.size() * ggml_xtc_mppa_elem_size(f16));
    ggml_xtc_mppa_write_elems(ctx, dev, v, f16);
    return dev;
}

// device F16 or F32 elements read as host floats
static void ggml_xtc_mppa_download_elems(const ggml_xtc_mppa_context & ctx, uintptr_t dev, std::vector<float> & v, bool f16) {
    if (f16) {
        std::vector<ggml_fp16_t> h(v.size());
        ctx.download(dev, h.data(), h.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(h.data(), v.data(), (int64_t) v.size());
    } else {
        ctx.download(dev, v.data(), v.size() * sizeof(float));
    }
}

// calls f once for the whole node when it is flat, else once per row of dst (see ggml_xtc_for_each_chunk)
template <typename F>
static void ggml_xtc_mppa_for_each_chunk(const struct ggml_tensor * dst, F && f) {
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

// RMS_NORM kernels as in ggml_xtc_run_rms_norm_kernels: x*x, row sums (matmul by 1/N), + eps, rsqrt,
// broadcast (matmul by ones), x * scale. Constants and temporaries live in the scratch of the node.
static void ggml_xtc_mppa_run_rms_norm(ggml_xtc_mppa_context & ctx, const std::vector<const ggml_xtc_mppa_kernel *> & fns,
                                       struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    const int64_t N = dst->ne[0];
    const int64_t M = ggml_xtc_flat_rows(dst) ? ggml_nrows(dst) : 1;
    const bool f16 = ggml_xtc_mppa_is_f16(dst);
    const size_t es = ggml_xtc_mppa_elem_size(f16);
    const std::vector<float> inv_n(N, 1.0f / (float) N), ones(N, 1.0f), eps_v(M, eps);

    ctx.reset();
    ctx.reserve(2 * ggml_xtc_mppa_align(N * es) + 4 * ggml_xtc_mppa_align(M * es) +
                2 * ggml_xtc_mppa_align(M * N * es));
    const uintptr_t d_inv_n    = ggml_xtc_mppa_upload_elems(ctx, inv_n, f16);
    const uintptr_t d_ones     = ggml_xtc_mppa_upload_elems(ctx, ones, f16);
    const uintptr_t d_eps      = ggml_xtc_mppa_upload_elems(ctx, eps_v, f16);
    const uintptr_t d_sq       = ctx.alloc(M * N * es);
    const uintptr_t d_mean     = ctx.alloc(M * es);
    const uintptr_t d_mean_eps = ctx.alloc(M * es);
    const uintptr_t d_rs       = ctx.alloc(M * es);
    const uintptr_t d_bc       = ctx.alloc(M * N * es);

    ggml_xtc_mppa_for_each_chunk(dst, [&](int64_t i1, int64_t i2, int64_t i3) {
        const uintptr_t x   = ggml_xtc_mppa_dev_addr(src0, ggml_xtc_mppa_row_off(src0, i1, i2, i3));
        const uintptr_t out = ggml_xtc_mppa_dev_addr(dst, ggml_xtc_mppa_row_off(dst, i1, i2, i3));
        ggml_xtc_mppa_invoke(*fns[0], { x, x, d_sq });
        ggml_xtc_mppa_invoke(*fns[1], { d_sq, d_inv_n, d_mean });
        ggml_xtc_mppa_invoke(*fns[2], { d_mean, d_eps, d_mean_eps });
        ggml_xtc_mppa_invoke(*fns[3], { d_mean_eps, d_rs });
        ggml_xtc_mppa_invoke(*fns[4], { d_rs, d_ones, d_bc });
        ggml_xtc_mppa_invoke(*fns[5], { x, d_bc, out });
    });
}

// SWIGLU kernels as in ggml_xtc_run_swiglu_kernels: silu(gate), then * up
static void ggml_xtc_mppa_run_swiglu(ggml_xtc_mppa_context & ctx, const std::vector<const ggml_xtc_mppa_kernel *> & fns,
                                     struct ggml_tensor * dst) {
    const struct ggml_tensor * gate = dst->src[0];
    const struct ggml_tensor * up   = dst->src[1];
    const int64_t N = dst->ne[0];
    const int64_t M = ggml_xtc_flat_rows(dst) ? ggml_nrows(dst) : 1;

    ctx.reset();
    ctx.reserve(ggml_xtc_mppa_align(M * N * ggml_xtc_mppa_elem_size(ggml_xtc_mppa_is_f16(dst))));
    const uintptr_t tmp = ctx.alloc(M * N * ggml_xtc_mppa_elem_size(ggml_xtc_mppa_is_f16(dst)));

    ggml_xtc_mppa_for_each_chunk(dst, [&](int64_t i1, int64_t i2, int64_t i3) {
        const uintptr_t g   = ggml_xtc_mppa_dev_addr(gate, ggml_xtc_mppa_row_off(gate, i1, i2, i3));
        const uintptr_t u   = ggml_xtc_mppa_dev_addr(up, ggml_xtc_mppa_row_off(up, i1, i2, i3));
        const uintptr_t out = ggml_xtc_mppa_dev_addr(dst, ggml_xtc_mppa_row_off(dst, i1, i2, i3));
        ggml_xtc_mppa_invoke(*fns[0], { g, tmp });
        ggml_xtc_mppa_invoke(*fns[1], { tmp, u, out });
    });
}

// ROPE kernels as in ggml_xtc_run_rope_kernels: matmul by the swap matrix, * sin, then x * cos + that.
// The swap matrix is uploaded once. cos and sin are rebuilt on the host for each token and written in place.
static void ggml_xtc_mppa_run_rope(ggml_xtc_mppa_context & ctx, const std::vector<const ggml_xtc_mppa_kernel *> & fns,
                                   struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * src2 = dst->src[2];

    xtc_rope_params p;
    p.n_dims = ggml_get_op_params_i32(dst, 1);
    p.neox   = ggml_get_op_params_i32(dst, 2) == GGML_ROPE_TYPE_NEOX;
    memcpy(&p.freq_base,   (const int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&p.freq_scale,  (const int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&p.attn_factor, (const int32_t *) dst->op_params + 8, sizeof(float));

    const int64_t D = dst->ne[0];
    const int64_t H = dst->ne[1];

    std::vector<int32_t> pos(src1->ne[0]);
    std::vector<float>   ff(src2 != nullptr ? src2->ne[0] : 0);
    ggml_xtc_mppa_tensor_read(src1, 0, pos.data(), pos.size() * sizeof(int32_t));
    if (src2 != nullptr) {
        ggml_xtc_mppa_tensor_read(src2, 0, ff.data(), ff.size() * sizeof(float));
    }

    const bool f16 = ggml_xtc_mppa_is_f16(dst);
    const size_t es = ggml_xtc_mppa_elem_size(f16);
    std::vector<float> swap(D * D), c(D), s(D), C(H * D), S(H * D);
    xtc_rope_swap_matrix(&p, D, swap.data());

    const size_t hd = H * D * es;
    ctx.reset();
    ctx.reserve(ggml_xtc_mppa_align(D * D * es) + 4 * ggml_xtc_mppa_align(hd));
    const uintptr_t d_swap = ggml_xtc_mppa_upload_elems(ctx, swap, f16);
    const uintptr_t t1     = ctx.alloc(hd);
    const uintptr_t t2     = ctx.alloc(hd);
    const uintptr_t d_C    = ctx.alloc(hd);
    const uintptr_t d_S    = ctx.alloc(hd);

    for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
            xtc_rope_factors(&p, D, pos[i2], src2 != nullptr ? ff.data() : nullptr, c.data(), s.data());
            for (int64_t h = 0; h < H; h++) {
                memcpy(&C[h * D], c.data(), D * sizeof(float));
                memcpy(&S[h * D], s.data(), D * sizeof(float));
            }
            ggml_xtc_mppa_write_elems(ctx, d_C, C, f16);
            ggml_xtc_mppa_write_elems(ctx, d_S, S, f16);

            // invoke is synchronous, so one slot of C and S is reused for all tokens
            const uintptr_t x   = ggml_xtc_mppa_dev_addr(src0, ggml_xtc_mppa_row_off(src0, 0, i2, i3));
            const uintptr_t out = ggml_xtc_mppa_dev_addr(dst, ggml_xtc_mppa_row_off(dst, 0, i2, i3));
            ggml_xtc_mppa_invoke(*fns[0], { x, d_swap, t1 });
            ggml_xtc_mppa_invoke(*fns[1], { t1, d_S, t2 });
            ggml_xtc_mppa_invoke(*fns[2], { x, d_C, t2, out });
        }
    }
}

// SET_ROWS (F16 dst) as in ggml_xtc_run_set_rows_kernels: one F32 -> F16 cast of each source row into dst row
// idx (indices are I64, read back to the host once). The index dims 1 and 2 broadcast over src dims 2 and 3.
static void ggml_xtc_mppa_run_set_rows(ggml_xtc_mppa_context & ctx, const std::vector<const ggml_xtc_mppa_kernel *> & fns,
                                       struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    const ggml_xtc_mppa_kernel & k = *fns[0];
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * idx  = dst->src[1];

    const int64_t n_idx_rows = idx->ne[1] * idx->ne[2];
    std::vector<int64_t> rows(n_idx_rows * idx->ne[0]);
    for (int64_t r = 0; r < n_idx_rows; r++) {
        const int64_t i2 = r % idx->ne[1];
        const int64_t i3 = r / idx->ne[1];
        ggml_xtc_mppa_tensor_read(idx, i2 * idx->nb[1] + i3 * idx->nb[2], &rows[r * idx->ne[0]],
                                  idx->ne[0] * sizeof(int64_t));
    }

    for (int64_t i3 = 0; i3 < src0->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < src0->ne[2]; i2++) {
            for (int64_t i = 0; i < src0->ne[1]; i++) {
                const int64_t r = (i3 % idx->ne[2]) * idx->ne[1] + (i2 % idx->ne[1]);
                const int64_t row = rows[r * idx->ne[0] + i];
                GGML_ASSERT(row >= 0 && row < dst->ne[1]);

                const uintptr_t a = ggml_xtc_mppa_dev_addr(src0, i * src0->nb[1] + i2 * src0->nb[2] + i3 * src0->nb[3]);
                const uintptr_t d = ggml_xtc_mppa_dev_addr(dst, row * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
                ggml_xtc_mppa_invoke(k, { a, d });
            }
        }
    }
}

// FLASH_ATTN_EXT kernels as in ggml_xtc_run_flash_attn_kernels, per (sequence, KV head). Q and the mask are
// gathered on the host and uploaded into fixed scratch slots. K and V are used in place when packed.
// The row max and row sum are fixed up on the host between the kernels.
static void ggml_xtc_mppa_run_flash_attn(ggml_xtc_mppa_context & ctx, const std::vector<const ggml_xtc_mppa_kernel *> & fns,
                                         struct ggml_tensor * dst) {
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

    const bool k_packed = k->nb[1] == D * sizeof(ggml_fp16_t);
    const bool v_packed = v->nb[1] == D * sizeof(ggml_fp16_t);

    const size_t qf_b  = M * D * sizeof(float);
    const size_t q16_b = M * D * sizeof(ggml_fp16_t);
    const size_t s_b   = M * n_kv * sizeof(float);
    const size_t p_b   = M * n_kv * sizeof(ggml_fp16_t);
    const size_t row_b = M * sizeof(float);
    const size_t kv_b  = n_kv * D * sizeof(ggml_fp16_t);

    std::vector<float>       qf(M * D), scale_v(M * n_kv, scale), mask_f(M * n_kv, 0.0f), row_max(M), row_sum(M), O(M * D);
    std::vector<ggml_fp16_t> mask_row(mask != nullptr ? n_kv : 0), k_g(k_packed ? 0 : n_kv * D), v_g(v_packed ? 0 : n_kv * D);

    ctx.reset();
    ctx.reserve(ggml_xtc_mppa_align(qf_b) + ggml_xtc_mppa_align(q16_b) + 4 * ggml_xtc_mppa_align(s_b) +
                ggml_xtc_mppa_align(p_b) + ggml_xtc_mppa_align(qf_b) + 2 * ggml_xtc_mppa_align(kv_b) +
                2 * ggml_xtc_mppa_align(row_b));
    const uintptr_t d_qf    = ctx.alloc(qf_b);
    const uintptr_t d_q16   = ctx.alloc(q16_b);
    const uintptr_t d_mask  = ctx.alloc(s_b);
    const uintptr_t d_scale = ctx.upload(scale_v.data(), s_b);
    const uintptr_t d_S     = ctx.alloc(s_b);
    const uintptr_t d_tmp   = ctx.alloc(s_b);
    const uintptr_t d_max   = ctx.alloc(row_b);
    const uintptr_t d_sum   = ctx.alloc(row_b);
    const uintptr_t d_p16   = ctx.alloc(p_b);
    const uintptr_t d_O     = ctx.alloc(qf_b);
    const uintptr_t d_k     = ctx.alloc(kv_b);
    const uintptr_t d_v     = ctx.alloc(kv_b);
    ctx.write(d_mask, mask_f.data(), s_b); // zeros when there is no mask

    // device address of a packed K or V head, or of its gathered copy in the slot dev
    auto kv_addr = [&](const struct ggml_tensor * t, bool packed, uintptr_t dev, std::vector<ggml_fp16_t> & host,
                       size_t off) {
        if (packed) {
            return ggml_xtc_mppa_dev_addr(t, off);
        }
        for (int64_t j = 0; j < n_kv; j++) {
            ggml_xtc_mppa_tensor_read(t, off + j * t->nb[1], &host[j * D], D * sizeof(ggml_fp16_t));
        }
        ctx.write(dev, host.data(), kv_b);
        return dev;
    };

    for (int64_t s = 0; s < q->ne[3]; s++) {
        for (int64_t g = 0; g < k->ne[2]; g++) {
            for (int64_t r = 0; r < M; r++) {
                const int64_t h = g * rk2 + r / n_tok;
                const int64_t t = r % n_tok;
                ggml_xtc_mppa_tensor_read(q, t * q->nb[1] + h * q->nb[2] + s * q->nb[3], &qf[r * D], D * sizeof(float));
                if (mask != nullptr) {
                    ggml_xtc_mppa_tensor_read(mask, t * mask->nb[1] + (h % mask->ne[2]) * mask->nb[2] +
                                              (s % mask->ne[3]) * mask->nb[3], mask_row.data(), n_kv * sizeof(ggml_fp16_t));
                    for (int64_t j = 0; j < n_kv; j++) {
                        mask_f[r * n_kv + j] = ggml_fp16_to_fp32(mask_row[j]);
                    }
                }
            }
            ctx.write(d_qf, qf.data(), qf_b);
            if (mask != nullptr) {
                ctx.write(d_mask, mask_f.data(), s_b);
            }

            const uintptr_t d_kk = kv_addr(k, k_packed, d_k, k_g, g * k->nb[2] + s * k->nb[3]);
            const uintptr_t d_vv = kv_addr(v, v_packed, d_v, v_g, g * v->nb[2] + s * v->nb[3]);

            ggml_xtc_mppa_invoke(*fns[0], { d_qf, d_q16 });                 // q16 = F16(Q)
            ggml_xtc_mppa_invoke(*fns[1], { d_q16, d_kk, d_S });            // S = q16.K^T
            ggml_xtc_mppa_invoke(*fns[2], { d_S, d_scale, d_mask, d_S });   // S = S * scale + mask
            ggml_xtc_mppa_invoke(*fns[3], { d_S, d_max });                  // row max
            ctx.download(d_max, row_max.data(), row_b);
            for (int64_t r = 0; r < M; r++) {
                if (row_max[r] == -INFINITY) { // fully masked row: exp(-inf) = 0 and a zero output
                    row_max[r] = 0.0f;
                }
            }
            ctx.write(d_max, row_max.data(), row_b);
            ggml_xtc_mppa_invoke(*fns[4], { d_S, d_max, d_tmp });           // S - max
            ggml_xtc_mppa_invoke(*fns[5], { d_tmp, d_tmp });                // exp
            ggml_xtc_mppa_invoke(*fns[6], { d_tmp, d_sum });                // row sum
            ctx.download(d_sum, row_sum.data(), row_b);
            for (int64_t r = 0; r < M; r++) {
                if (row_sum[r] == 0.0f) {
                    row_sum[r] = 1.0f;
                }
            }
            ctx.write(d_sum, row_sum.data(), row_b);
            ggml_xtc_mppa_invoke(*fns[7], { d_tmp, d_sum, d_tmp });         // P
            ggml_xtc_mppa_invoke(*fns[8], { d_tmp, d_p16 });                // F16(P)
            ggml_xtc_mppa_invoke(*fns[9], { d_p16, d_vv, d_O });            // out = P.V
            ctx.download(d_O, O.data(), qf_b);

            for (int64_t r = 0; r < M; r++) {
                const int64_t h = g * rk2 + r / n_tok;
                const int64_t t = r % n_tok;
                ggml_xtc_mppa_tensor_write(dst, h * dst->nb[1] + t * dst->nb[2] + s * dst->nb[3], &O[r * D], D * sizeof(float));
            }
        }
    }
}

// F16 FLASH_ATTN_EXT (ggml_xtc_node_kernels_f16): as above, but Q is already F16 and the scores, the softmax and P
// stay F16 on the device; the F32 output is converted on the host. Row max and sum are fixed up on the host.
static void ggml_xtc_mppa_run_flash_attn_f16(ggml_xtc_mppa_context & ctx, const std::vector<const ggml_xtc_mppa_kernel *> & fns,
                                             struct ggml_tensor * dst) {
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

    const bool k_packed = k->nb[1] == D * sizeof(ggml_fp16_t);
    const bool v_packed = v->nb[1] == D * sizeof(ggml_fp16_t);

    const size_t q_b   = M * D * sizeof(ggml_fp16_t);
    const size_t s_b   = M * n_kv * sizeof(ggml_fp16_t);
    const size_t row_b = M * sizeof(ggml_fp16_t);
    const size_t kv_b  = n_kv * D * sizeof(ggml_fp16_t);

    std::vector<float>       scale_v(M * n_kv, scale), row_max(M), row_sum(M), O(M * D);
    std::vector<ggml_fp16_t> q_h(M * D), mask_h(M * n_kv, ggml_fp32_to_fp16(0.0f)), o_h(M * D);
    std::vector<ggml_fp16_t> k_g(k_packed ? 0 : n_kv * D), v_g(v_packed ? 0 : n_kv * D);

    ctx.reset();
    ctx.reserve(2 * ggml_xtc_mppa_align(q_b) + 4 * ggml_xtc_mppa_align(s_b) + 2 * ggml_xtc_mppa_align(kv_b) +
                2 * ggml_xtc_mppa_align(row_b));
    const uintptr_t d_q     = ctx.alloc(q_b);
    const uintptr_t d_mask  = ctx.alloc(s_b);
    const uintptr_t d_scale = ggml_xtc_mppa_upload_elems(ctx, scale_v, true);
    const uintptr_t d_S     = ctx.alloc(s_b);
    const uintptr_t d_tmp   = ctx.alloc(s_b);
    const uintptr_t d_max   = ctx.alloc(row_b);
    const uintptr_t d_sum   = ctx.alloc(row_b);
    const uintptr_t d_O     = ctx.alloc(q_b);
    const uintptr_t d_k     = ctx.alloc(kv_b);
    const uintptr_t d_v     = ctx.alloc(kv_b);
    ctx.write(d_mask, mask_h.data(), s_b); // zeros when there is no mask

    auto kv_addr = [&](const struct ggml_tensor * t, bool packed, uintptr_t dev, std::vector<ggml_fp16_t> & host,
                       size_t off) {
        if (packed) {
            return ggml_xtc_mppa_dev_addr(t, off);
        }
        for (int64_t j = 0; j < n_kv; j++) {
            ggml_xtc_mppa_tensor_read(t, off + j * t->nb[1], &host[j * D], D * sizeof(ggml_fp16_t));
        }
        ctx.write(dev, host.data(), kv_b);
        return dev;
    };

    for (int64_t s = 0; s < q->ne[3]; s++) {
        for (int64_t g = 0; g < k->ne[2]; g++) {
            for (int64_t r = 0; r < M; r++) {
                const int64_t h = g * rk2 + r / n_tok;
                const int64_t t = r % n_tok;
                ggml_xtc_mppa_tensor_read(q, t * q->nb[1] + h * q->nb[2] + s * q->nb[3], &q_h[r * D], D * sizeof(ggml_fp16_t));
                if (mask != nullptr) {
                    ggml_xtc_mppa_tensor_read(mask, t * mask->nb[1] + (h % mask->ne[2]) * mask->nb[2] +
                                              (s % mask->ne[3]) * mask->nb[3], &mask_h[r * n_kv], n_kv * sizeof(ggml_fp16_t));
                }
            }
            ctx.write(d_q, q_h.data(), q_b);
            if (mask != nullptr) {
                ctx.write(d_mask, mask_h.data(), s_b);
            }

            const uintptr_t d_kk = kv_addr(k, k_packed, d_k, k_g, g * k->nb[2] + s * k->nb[3]);
            const uintptr_t d_vv = kv_addr(v, v_packed, d_v, v_g, g * v->nb[2] + s * v->nb[3]);

            ggml_xtc_mppa_invoke(*fns[0], { d_q, d_kk, d_S });              // S = Q.K^T
            ggml_xtc_mppa_invoke(*fns[1], { d_S, d_scale, d_mask, d_S });   // S = S * scale + mask
            ggml_xtc_mppa_invoke(*fns[2], { d_S, d_max });                  // row max
            ggml_xtc_mppa_download_elems(ctx, d_max, row_max, true);
            for (int64_t r = 0; r < M; r++) {
                if (row_max[r] == -INFINITY) { // fully masked row: exp(-inf) = 0 and a zero output
                    row_max[r] = 0.0f;
                }
            }
            ggml_xtc_mppa_write_elems(ctx, d_max, row_max, true);
            ggml_xtc_mppa_invoke(*fns[3], { d_S, d_max, d_tmp });           // S - max
            ggml_xtc_mppa_invoke(*fns[4], { d_tmp, d_tmp });                // exp
            ggml_xtc_mppa_invoke(*fns[5], { d_tmp, d_sum });                // row sum
            ggml_xtc_mppa_download_elems(ctx, d_sum, row_sum, true);
            for (int64_t r = 0; r < M; r++) {
                if (row_sum[r] == 0.0f) {
                    row_sum[r] = 1.0f;
                }
            }
            ggml_xtc_mppa_write_elems(ctx, d_sum, row_sum, true);
            ggml_xtc_mppa_invoke(*fns[6], { d_tmp, d_sum, d_tmp });         // P
            ggml_xtc_mppa_invoke(*fns[7], { d_tmp, d_vv, d_O });            // out = P.V
            ctx.download(d_O, o_h.data(), q_b);
            ggml_fp16_to_fp32_row(o_h.data(), O.data(), (int64_t) O.size());

            for (int64_t r = 0; r < M; r++) {
                const int64_t h = g * rk2 + r / n_tok;
                const int64_t t = r % n_tok;
                ggml_xtc_mppa_tensor_write(dst, h * dst->nb[1] + t * dst->nb[2] + s * dst->nb[3], &O[r * D], D * sizeof(float));
            }
        }
    }
}

static enum ggml_status ggml_backend_xtc_mppa_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(g_xtc_mppa_mutex);
    g_xtc_mppa_generation++;
    auto & ctx = *static_cast<ggml_xtc_mppa_context *>(backend->context);
    ctx.reset();

    // pass 1: resolve every kernel and check the tensors *before* computing anything
    std::vector<std::vector<const ggml_xtc_mppa_kernel *>> fns(cgraph->n_nodes);
    std::vector<std::string> errors;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_op_is_empty(node->op)) {
            continue;
        }
        const std::vector<ggml_xtc_kernel_shape> shapes = ggml_xtc_mppa_node_kernels(node);
        if (shapes.empty() || !ggml_xtc_mppa_has_runner(node)) {
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
        for (const ggml_xtc_kernel_shape & shape : shapes) {
            std::string err;
            const ggml_xtc_mppa_kernel * k = ggml_xtc_mppa_get_kernel(shape, err);
            if (k == nullptr) {
                errors.push_back(err);
            } else {
                fns[i].push_back(k);
            }
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
        if (fns[i].empty()) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_xtc_mppa_run_matmul(ctx, fns[i], node);
                break;
            case GGML_OP_ADD:
            case GGML_OP_MUL:
                ggml_xtc_mppa_run_add_mul(ctx, fns[i], node);
                break;
            case GGML_OP_RMS_NORM:
                ggml_xtc_mppa_run_rms_norm(ctx, fns[i], node);
                break;
            case GGML_OP_GLU:
                ggml_xtc_mppa_run_swiglu(ctx, fns[i], node);
                break;
            case GGML_OP_ROPE:
                ggml_xtc_mppa_run_rope(ctx, fns[i], node);
                break;
            case GGML_OP_SET_ROWS:
                ggml_xtc_mppa_run_set_rows(ctx, fns[i], node);
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                if (ggml_xtc_mppa_is_f16(node)) {
                    ggml_xtc_mppa_run_flash_attn_f16(ctx, fns[i], node);
                } else {
                    ggml_xtc_mppa_run_flash_attn(ctx, fns[i], node);
                }
                break;
            case GGML_OP_CPY:
                ggml_xtc_mppa_invoke(*fns[i][0], { ggml_xtc_mppa_dev_addr(node->src[0]), ggml_xtc_mppa_dev_addr(node) });
                break;
            default:
                GGML_ABORT("%s: unexpected op %s", __func__, ggml_op_name(node->op));
        }
    }

    return GGML_STATUS_SUCCESS;
}

static const char * ggml_backend_xtc_mppa_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_XTC_MPPA_NAME;
}

static void ggml_backend_xtc_mppa_free(ggml_backend_t backend) {
    delete static_cast<ggml_xtc_mppa_context *>(backend->context);
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
        /* .context = */ new ggml_xtc_mppa_context(),
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
    return ggml_xtc_mppa_supports(op);
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
