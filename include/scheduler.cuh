// CUDA runtime
#include <cuda_runtime.h>
#include <stdlib.h>

#include <memory>

#include "utility.cuh"

#ifndef SCHEDULER
#define SCHEDULER

constexpr size_t MAX_ARGS_SZ = 128;

class KernelSlice;
class KernelConfig;

typedef FixSizeBox<MAX_ARGS_SZ> Args;
typedef void (*KernelPtr)(Args, KernelSlice);
typedef int KernelSliceID;

class KernelArgs {};

class KernelSlice {
 private:
  dim3 grid_dim;
  KernelSliceID kernel_slice_id;

 public:
  KernelSlice(dim3 grid_dim, KernelSliceID kernel_slice_id)
      : grid_dim(grid_dim), kernel_slice_id(kernel_slice_id) {}
  __device__ __host__ dim3 get_original_block_idx() {
    int x = kernel_slice_id % grid_dim.x;
    int z = kernel_slice_id / (grid_dim.x * grid_dim.y);
    int y = (kernel_slice_id - z * grid_dim.x * grid_dim.y) / grid_dim.x;

    return dim3(x, y, z);
  }
};

class KernelConfig {
  KernelPtr kernel_ptr;
  Args args;
  dim3 grid_dim;
  dim3 block_dim;

 public:
  KernelConfig(const KernelConfig&) = default;
  KernelConfig(KernelConfig&&) = default;

  template <class Arg_Ty>
  KernelConfig(KernelPtr kernel_ptr, Arg_Ty args, dim3 grid_dim, dim3 block_dim)
      : kernel_ptr(kernel_ptr),
        args(args),
        grid_dim(grid_dim),
        block_dim(block_dim) {}

  template <class Arg_Ty>
  Arg_Ty* get_args() {
    return args.as<Arg_Ty>();
  }

  size_t get_block_num() const { return grid_dim.x * grid_dim.y * grid_dim.z; }

  auto __device__ __host__ get_grid_dim() const { return grid_dim; }

  void launch(KernelSliceID kernel_slice_id) {
    kernel_ptr<<<grid_dim, block_dim>>>(args,
                                        KernelSlice{grid_dim, kernel_slice_id});
  }
};

class KernelScheduler {
 public:
};

#endif