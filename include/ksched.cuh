// CUDA runtime
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdlib.h>

#include <memory>

#include "utility.cuh"

#ifndef SCHEDULER
#define SCHEDULER

constexpr size_t MAX_ARGS_SZ = 128;
constexpr size_t MAX_CONTEXT_SZ = 128;

class KernelSlice;
class KernelConfig;
class Kernel;

typedef FixSizeBox<MAX_ARGS_SZ> Args;
typedef FixSizeBox<MAX_CONTEXT_SZ> Context;
typedef void (*KernelPtr)(Args, KernelSlice);
typedef Range<unsigned> KernelSliceRange;

using PreProcessFunc = KernelConfig (*)();
using PostProcessFunc = void (*)(Kernel&);
using ExecuteFunc = void (*)(KernelSliceRange kernel_slice_range,
                             dim3 block_dim, size_t shared_mem,
                             cudaStream_t stream, dim3 grid_dim, Args args);

/**
 * C++ name mangling
 */
const char* PREPROCESS_FUNC_NAME = "pre_process";
const char* POSTPROCESS_FUNC_NAME = "post_process";
const char* EXECUTE_FUNC_NAME = "execute";

/**
 * Functions defined by the users:
 * 1. pre_process
 * 2. execute
 * 3. post_process
 */
#define DEFAULT_EXECUTE(KERNEL_NAME)                                          \
  EXPORT                                                                      \
  void execute(KernelSliceRange kernel_slice_range, dim3 block_dim,           \
               size_t shared_mem, cudaStream_t stream, dim3 grid_dim,         \
               Args args) {                                                   \
    KERNEL_NAME<<<kernel_slice_range.len(), block_dim, shared_mem, stream>>>( \
        args, KernelSlice{grid_dim, kernel_slice_range});                     \
  }

class KernelSlice {
 private:
  dim3 grid_dim;
  KernelSliceRange kernel_slice_range;

 public:
  KernelSlice(dim3 grid_dim, KernelSliceRange kernel_slice_range)
      : grid_dim(grid_dim), kernel_slice_range(kernel_slice_range) {
    if (!KernelSliceRange{0, grid_dim.x * grid_dim.y * grid_dim.z}.contains(
            kernel_slice_range)) {
      fprintf(stderr, "kernel slice range %d %d\n", kernel_slice_range.low(),
              kernel_slice_range.high());
      ERR("Out of grid size");
    }
  }
  __device__ dim3 get_original_block_idx() {
    if (gridDim.y != 1 || gridDim.z != 1) {
      ERR("Kernel slice should be 1-dimensional");
    }
    int block_uid = blockIdx.x + kernel_slice_range.low();
    if (block_uid >= kernel_slice_range.high()) {
      ERR("out of kernel slice");
    }
    int x = block_uid % grid_dim.x;
    int z = block_uid / (grid_dim.x * grid_dim.y);
    int y = (block_uid - z * grid_dim.x * grid_dim.y) / grid_dim.x;

    return dim3(x, y, z);
  }
};

struct KernelConfig {
  KernelPtr kernel_ptr;
  Args args;
  dim3 grid_dim;
  dim3 block_dim;
  Context context;
};

class Kernel {
 private:
  void* kernel_lib_handler;
  KernelPtr kernel_ptr;
  Args args;

  // Shape
  dim3 grid_dim;
  dim3 block_dim;

  Context context;

  // Characteristics
  // TODO

 public:
  // Kernel(const Kernel&) = default;
  // Kernel(Kernel&&) = default;

  Kernel(const char* kernel_lib_file_path) {
    kernel_lib_handler = dlopen(kernel_lib_file_path, RTLD_NOW);
    if (!kernel_lib_handler) {
      printf("Error loading kernel lib %s:%s\n", kernel_lib_file_path,
             dlerror());
    }
  }
  ~Kernel() {
    if (kernel_lib_handler) dlclose(kernel_lib_handler);
  }

  void pre_process() {
    PreProcessFunc pre_process = reinterpret_cast<PreProcessFunc>(
        dlsym(kernel_lib_handler, PREPROCESS_FUNC_NAME));
    if (!pre_process) {
      ERR("Error linking function pre_process");
      return;
    }
    KernelConfig config = pre_process();
    kernel_ptr = config.kernel_ptr, args = config.args,
    grid_dim = config.grid_dim, block_dim = config.block_dim,
    context = config.context;
  }

  void post_process() {
    PostProcessFunc post_process = reinterpret_cast<PostProcessFunc>(
        dlsym(kernel_lib_handler, POSTPROCESS_FUNC_NAME));
    if (!post_process) {
      ERR("Error linking function post_process");
    }
    post_process(*this);
  }

  template <class Arg_Ty>
  Arg_Ty* get_args() {
    return args.as<Arg_Ty>();
  }

  template <class Context_Ty>
  Context_Ty* get_context() {
    return context.as<Context_Ty>();
  }

  size_t get_block_num() const { return grid_dim.x * grid_dim.y * grid_dim.z; }

  auto __device__ __host__ get_grid_dim() const { return grid_dim; }

  void launch(KernelSliceRange kernel_slice_range) {
    kernel_ptr<<<kernel_slice_range.len(), block_dim>>>(
        args, KernelSlice{grid_dim, kernel_slice_range});
  }

  void launch(KernelSliceRange kernel_slice_range, size_t shared_mem_size) {
    kernel_ptr<<<kernel_slice_range.len(), block_dim, shared_mem_size>>>(
        args, KernelSlice{grid_dim, kernel_slice_range});
  }

  void launch(KernelSliceRange kernel_slice_range, cudaStream_t stream) {
    auto execute =
        reinterpret_cast<ExecuteFunc>(dlsym(kernel_lib_handler, "execute"));
    if (!execute) {
      ERR("Error linking function execute");
    }
    execute(kernel_slice_range, block_dim, 0, stream, grid_dim, args);
    // kernel_ptr<<<kernel_slice_range.len(), block_dim, 0, stream>>>(
    //     args, KernelSlice{grid_dim, kernel_slice_range});
  }

  void launch(KernelSliceRange kernel_slice_range, size_t shared_mem_size,
              cudaStream_t stream) {
    kernel_ptr<<<kernel_slice_range.len(), block_dim, shared_mem_size,
                 stream>>>(args, KernelSlice{grid_dim, kernel_slice_range});
  }
};

#endif