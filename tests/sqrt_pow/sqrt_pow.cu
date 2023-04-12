
// System includes
#include <math.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

#include "kosched.cuh"
#include "utility.cuh"

struct SqrtPowArgs {
  double *data;
  long n;
};

__global__ void sqrt_pow(Args args, KernelSlice kernel_slice) {
  auto sp_args = args.as<SqrtPowArgs>();
  double *x = sp_args->data;
  long n = sp_args->n;

  /****************************************************************/
  // rebuild blockId
  dim3 blockIdx = kernel_slice.get_original_block_idx();
  dim3 gridIdx = kernel_slice.get_original_grid_idx();
  /****************************************************************/

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  //   for (long i = tid; i < n; i += blockDim.x * gridDim.x) {
  //     x[i] = sqrt(pow(3.14159, (double)x[i]));
  //   }
  for (int i = 0; i < n; i++) {
    x[tid] = sqrt(pow(3.14159, (double)x[tid]));
  }
}

EXPORT KernelConfig pre_process() {
  dim3 block_dim = dim3(32);
  dim3 grid_dim = dim3(1 << 11);

  size_t mem_size = sizeof(double) * grid_dim.x * block_dim.x;

  double *data = (double *)malloc(mem_size);
  for (size_t i = 0; i < grid_dim.x * block_dim.x; i++) {
    data[i] = i;
  }

  double *d_data;
  CHECK(cudaMalloc((void **)&d_data, mem_size));

  CHECK(cudaMemcpy(d_data, data, mem_size, cudaMemcpyHostToDevice));
  SqrtPowArgs args{d_data, 1 << 6};
  return KernelConfig{(KernelPtr)sqrt_pow, Args{args}, grid_dim, block_dim,
                      Context{data}};
}

DEFAULT_EXECUTE(sqrt_pow)

EXPORT void post_process(Kernel &kernel) {
  CHECK(cudaFree(kernel.get_args<SqrtPowArgs>()->data));
  free(*kernel.get_context<double *>());
}
