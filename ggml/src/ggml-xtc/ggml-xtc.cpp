#include "ggml-xtc.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
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

struct ggml_backend_xtc_context {
    ggml_backend_t cpu = nullptr;
    int            n_threads = GGML_DEFAULT_N_THREADS;
};

using ggml_xtc_kernel_fn = void (*)(void *, void *, void *);

struct ggml_xtc_kernel_desc {
    enum ggml_op   op;
    enum ggml_type type0;
    enum ggml_type type1;
    int64_t        ne0  = 0;
    int64_t        ne1  = 0;
    int64_t        ne10 = 0;
    int64_t        ne02 = 1;
    int64_t        ne03 = 1;
    int64_t        ne12 = 1;
    int64_t        ne13 = 1;
    ggml_xtc_kernel_fn func = nullptr;
};

struct ggml_xtc_kernel_library {
#if defined(_WIN32)
    HMODULE handle = nullptr;
#else
    void * handle = nullptr;
#endif
    std::string path;
};

static std::mutex                              g_xtc_kernels_mutex;
static std::vector<ggml_xtc_kernel_desc>       g_xtc_kernels;
static std::vector<ggml_xtc_kernel_library>    g_xtc_kernel_libs;
static bool                                    g_xtc_kernels_loaded = false;

static bool ggml_xtc_is_disabled() {
    const char * env = getenv("GGML_XTC_DISABLE");
    return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0;
}

static enum ggml_type ggml_xtc_type_from_name(const char * name) {
    if (strcmp(name, "F32") == 0) {
        return GGML_TYPE_F32;
    }
    if (strcmp(name, "F16") == 0) {
        return GGML_TYPE_F16;
    }
    if (strcmp(name, "BF16") == 0) {
        return GGML_TYPE_BF16;
    }
    return GGML_TYPE_COUNT;
}

static enum ggml_op ggml_xtc_op_from_name(const char * name) {
    if (strcmp(name, "MUL_MAT") == 0) {
        return GGML_OP_MUL_MAT;
    }
    return GGML_OP_NONE;
}

static void * ggml_xtc_load_sym(void * handle, const char * sym) {
#if defined(_WIN32)
    return (void *) GetProcAddress((HMODULE) handle, sym);
#else
    return dlsym(handle, sym);
#endif
}

static void * ggml_xtc_dlopen(const char * path) {
#if defined(_WIN32)
    return (void *) LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void ggml_xtc_dlclose(void * handle) {
    if (handle == nullptr) {
        return;
    }
#if defined(_WIN32)
    FreeLibrary((HMODULE) handle);
#else
    dlclose(handle);
#endif
}

static void ggml_xtc_load_kernels_from_dir(const fs::path & kernel_dir) {
    const fs::path manifest = kernel_dir / "manifest.csv";
    if (!fs::exists(manifest)) {
        return;
    }

    std::ifstream in(manifest);
    if (!in) {
        GGML_LOG_ERROR("%s: failed to open %s\n", __func__, manifest.string().c_str());
        return;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        char op_name[32];
        char type0_name[16];
        char type1_name[16];
        char lib_name[256];
        char sym_name[256];
        int64_t ne0;
        int64_t ne1;
        int64_t ne10;
        int64_t ne02;
        int64_t ne03;
        int64_t ne12;
        int64_t ne13;

        if (sscanf(line.c_str(), " %31[^,],%15[^,],%15[^,],%lld,%lld,%lld,%lld,%lld,%lld,%lld,%255[^,],%255s",
                   op_name, type0_name, type1_name,
                   (long long *) &ne0, (long long *) &ne1, (long long *) &ne10,
                   (long long *) &ne02, (long long *) &ne03, (long long *) &ne12, (long long *) &ne13,
                   lib_name, sym_name) < 12) {
            GGML_LOG_ERROR("%s: invalid manifest line: %s\n", __func__, line.c_str());
            continue;
        }

        const enum ggml_op   op    = ggml_xtc_op_from_name(op_name);
        const enum ggml_type type0 = ggml_xtc_type_from_name(type0_name);
        const enum ggml_type type1 = ggml_xtc_type_from_name(type1_name);
        if (op == GGML_OP_NONE || type0 == GGML_TYPE_COUNT || type1 == GGML_TYPE_COUNT) {
            GGML_LOG_ERROR("%s: unsupported manifest entry: %s\n", __func__, line.c_str());
            continue;
        }

        const fs::path lib_path = kernel_dir / lib_name;
        void * handle = nullptr;
        for (const auto & entry : g_xtc_kernel_libs) {
            if (entry.path == lib_path.string()) {
                handle = entry.handle;
                break;
            }
        }
        if (handle == nullptr) {
            handle = ggml_xtc_dlopen(lib_path.string().c_str());
            if (handle == nullptr) {
#if defined(_WIN32)
                GGML_LOG_ERROR("%s: failed to load %s\n", __func__, lib_path.string().c_str());
#else
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, lib_path.string().c_str(), dlerror());
#endif
                continue;
            }
            g_xtc_kernel_libs.push_back({ handle, lib_path.string() });
        }

        ggml_xtc_kernel_fn fn = (ggml_xtc_kernel_fn) ggml_xtc_load_sym(handle, sym_name);
        if (fn == nullptr) {
            GGML_LOG_ERROR("%s: symbol %s not found in %s\n", __func__, sym_name, lib_path.string().c_str());
            continue;
        }

        g_xtc_kernels.push_back({
            op, type0, type1, ne0, ne1, ne10, ne02, ne03, ne12, ne13, fn,
        });
    }
}

static void ggml_xtc_ensure_kernels_loaded() {
    std::lock_guard<std::mutex> lock(g_xtc_kernels_mutex);
    if (g_xtc_kernels_loaded) {
        return;
    }
    g_xtc_kernels_loaded = true;

    const char * kernel_dir = getenv("GGML_XTC_KERNEL_DIR");
    if (kernel_dir == nullptr || kernel_dir[0] == '\0') {
        return;
    }

    ggml_xtc_load_kernels_from_dir(fs::path(kernel_dir));
}

static bool ggml_xtc_mul_mat_dims_match(const struct ggml_tensor * op, const ggml_xtc_kernel_desc & k) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    return op->ne[0] == k.ne0 &&
           op->ne[1] == k.ne1 &&
           src1->ne[0] == k.ne10 &&
           src0->ne[2] == k.ne02 &&
           src0->ne[3] == k.ne03 &&
           src1->ne[2] == k.ne12 &&
           src1->ne[3] == k.ne13;
}

static const ggml_xtc_kernel_desc * ggml_xtc_find_kernel(const struct ggml_tensor * op) {
    ggml_xtc_ensure_kernels_loaded();

    for (const auto & k : g_xtc_kernels) {
        if (k.op != op->op) {
            continue;
        }
        if (op->op == GGML_OP_MUL_MAT) {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            if (src0->type != k.type0 || src1->type != k.type1) {
                continue;
            }
            if (!ggml_xtc_mul_mat_dims_match(op, k)) {
                continue;
            }
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                continue;
            }
            return &k;
        }
    }

    return nullptr;
}

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

static void ggml_xtc_run_mul_mat_kernel(const ggml_xtc_kernel_desc & k, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne0 == k.ne0);
    GGML_ASSERT(ne1 == k.ne1);
    GGML_ASSERT(ne10 == k.ne10);
    GGML_ASSERT(ne02 == k.ne02 && ne03 == k.ne03);
    GGML_ASSERT(ne12 == k.ne12 && ne13 == k.ne13);

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;

            void * a = (void *) ((char *) src1->data + i12 * nb12 + i13 * nb13);
            void * b = (void *) ((char *) src0->data + i02 * nb02 + i03 * nb03);
            void * c = (void *) ((char *) dst->data  + i12 * nb2  + i13 * nb3);

            k.func(a, b, c);
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

    if (const ggml_xtc_kernel_desc * k = ggml_xtc_find_kernel(op)) {
        GGML_UNUSED(k);
        return true;
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

    int cpu_i0 = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_xtc_node_is_noop(node)) {
            continue;
        }

        const ggml_xtc_kernel_desc * kernel = ggml_xtc_find_kernel(node);
        if (kernel == nullptr) {
            continue;
        }

        enum ggml_status status = ggml_xtc_flush_cpu_segment(ctx, cgraph, cpu_i0, i);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_xtc_run_mul_mat_kernel(*kernel, node);
                break;
            default:
                GGML_ABORT("%s: unsupported XTC op %s\n", __func__, ggml_op_desc(node));
        }

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

    ggml_xtc_ensure_kernels_loaded();

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
