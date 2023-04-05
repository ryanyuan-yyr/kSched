#include <cuda_runtime.h>

#include <algorithm>

#include "ksched.cuh"

__host__ int main() {
  Kernel kernel{"build/matrix_mul.so"};
  kernel.pre_process();
  constexpr int NSTREAM = 4;
  cudaStream_t streams[NSTREAM];
  for (size_t i = 0; i < NSTREAM; i++) {
    cudaStreamCreate(streams + i);
  }

  constexpr int step = 3;
  unsigned nblock = kernel.get_block_num();

  printf("nblock %d\n", nblock);

  for (unsigned i = 0; i < nblock; i += step) {
    kernel.launch(KernelSliceRange{i, std::min(i + step, nblock)},
                  streams[i % NSTREAM]);
  }

  CHECK(cudaDeviceSynchronize());

  kernel.post_process();
}