#include <cuda_runtime.h>

#include <algorithm>

#include "ksched.cuh"

__host__ int main() {
  // set up max connectioin
  const char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
  setenv(iname, "32", 1);

  Kernel matrix_mul_kernel{"build/matrix_mul.so"};
  Kernel matrix_transpose_kernel{"build/matrix_transpose.so"};

  constexpr int NSTREAM = 8;
  cudaStream_t streams[NSTREAM];
  for (size_t i = 0; i < NSTREAM; i++) {
    cudaStreamCreate(streams + i);
  }

  constexpr int mm_step = 3, mt_step = 24;
  double start, end;
  unsigned mm_nblock, mt_nblock;

  // warmup
  matrix_mul_kernel.pre_process();
  matrix_transpose_kernel.pre_process();
  mm_nblock = matrix_mul_kernel.get_block_num();
  mt_nblock = matrix_transpose_kernel.get_block_num();

  // for (unsigned i = 0; i < mm_nblock; i += mm_step) {
  //   // if (i < mm_nblock)
  matrix_mul_kernel.launch(KernelSliceRange{0, mm_nblock}, streams[0]);
  // }

  // for (unsigned i = 0; i < mt_nblock; i += mt_step) {
  // if (i < mt_nblock)
  matrix_transpose_kernel.launch(KernelSliceRange{0, mt_nblock}, streams[1]);
  // }
  CHECK(cudaDeviceSynchronize());
  matrix_mul_kernel.post_process();
  matrix_transpose_kernel.post_process();

  // Mix
  matrix_mul_kernel.pre_process();
  matrix_transpose_kernel.pre_process();
  mm_nblock = matrix_mul_kernel.get_block_num();
  mt_nblock = matrix_transpose_kernel.get_block_num();

  start = current_seconds();

  for (unsigned i = 0, j = 0, iter = 0; i < mm_nblock || j < mt_nblock;
       iter++) {
    if (i < mm_nblock) {
      matrix_mul_kernel.launch(
          KernelSliceRange{i, std::min(i + mm_step, mm_nblock)},
          streams[iter % (NSTREAM / 2)]);
      i += mm_step;
    }
    for (int mt_iter = 0; mt_iter < 30; mt_iter++) {
      if (j < mt_nblock) {
        matrix_transpose_kernel.launch(
            KernelSliceRange{j, std::min(j + mt_step, mt_nblock)},
            (streams + NSTREAM / 2)[iter % (NSTREAM / 2)]);
        j += mt_step;
      }
    }
    // if (iter % 16 == 0) {
    //   CHECK(cudaDeviceSynchronize());
    // }
  }

  CHECK(cudaDeviceSynchronize());

  end = current_seconds();

  printf("============================== Mix Duration %lf\n", end - start);

  matrix_mul_kernel.post_process();
  matrix_transpose_kernel.post_process();

  // not sliced
  matrix_mul_kernel.pre_process();
  matrix_transpose_kernel.pre_process();
  mm_nblock = matrix_mul_kernel.get_block_num();
  mt_nblock = matrix_transpose_kernel.get_block_num();

  start = current_seconds();

  matrix_mul_kernel.launch(KernelSliceRange{0, mm_nblock}, (cudaStream_t)NULL);
  matrix_transpose_kernel.launch(KernelSliceRange{0, mt_nblock}, (cudaStream_t)NULL);
  CHECK(cudaDeviceSynchronize());

  end = current_seconds();

  printf("Not ============================== sliced Duration %lf\n", end - start);

  matrix_mul_kernel.post_process();
  matrix_transpose_kernel.post_process();

  // serial
  matrix_mul_kernel.pre_process();
  matrix_transpose_kernel.pre_process();
  mm_nblock = matrix_mul_kernel.get_block_num();
  mt_nblock = matrix_transpose_kernel.get_block_num();

  start = current_seconds();

  for (unsigned i = 0, iter = 0; i < mm_nblock; i += mm_step, iter++) {
    // if (i < mm_nblock)
    matrix_mul_kernel.launch(
        KernelSliceRange{i, std::min(i + mm_step, mm_nblock)},
        streams[iter % (NSTREAM)]);
  }

  for (unsigned i = 0, iter = 0; i < mt_nblock; i += mt_step, iter++) {
    // if (i < mt_nblock)
    matrix_transpose_kernel.launch(
        KernelSliceRange{i, std::min(i + mt_step, mt_nblock)},
        streams[iter % NSTREAM]);
  }
  CHECK(cudaDeviceSynchronize());

  end = current_seconds();

  printf("============================== Serial Duration %lf\n", end - start);

  matrix_mul_kernel.post_process();
  matrix_transpose_kernel.post_process();
}