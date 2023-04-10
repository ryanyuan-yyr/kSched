#include <cuda_runtime.h>

#include <algorithm>

#include "ksched.cuh"

/**
 * TODO repair
 */
// bool can_cosched(Kernel& first, Kernel& second) {
//   int max_nthread_per_sm = 2048;
//   if (max_nthread_per_sm -
//           first.get_block_num() * first.get_nthread_per_block() >=
//       (max_nthread_per_sm -
//        (first.get_block_num() - 1) * first.get_nthread_per_block()) /
//           second.get_nthread_per_block() * second.get_nthread_per_block())
//     return true;
//   return false;
// }

const char* bool_to_str(bool value) { return value ? "true" : "false"; }

__host__ int main() {
  // set up max connectioin
  const char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
  setenv(iname, "32", 1);

  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);

  printf(
      "Number of SM %d; Number of cores per SM: %d; Max threads per SM %d; Max "
      "thread per block %d; Max reg per block %d\n",
      devProp.multiProcessorCount, get_ncore_pSM(devProp),
      devProp.maxThreadsPerMultiProcessor, devProp.maxThreadsPerBlock,
      devProp.regsPerBlock);

  Kernel vec_add_kernel{"build/vec_add.so"};
  Kernel sqrt_pow_kernel{"build/sqrt_pow.so"};

  constexpr int NSTREAM = 2;
  cudaStream_t streams[NSTREAM];
  for (size_t i = 0; i < NSTREAM; i++) {
    cudaStreamCreate(streams + i);
  }

  constexpr int mm_step = 3 * 1 * 764, mt_step = 3 * 4 * 12;

  // Motivation test config
  // constexpr int mm_step = 3 * 1 * 128 * 8, mt_step = 3 * 4 * 4 * 1;

  double start, end;
  unsigned mm_nblock, mt_nblock;

  // warmup
  vec_add_kernel.pre_process();
  sqrt_pow_kernel.pre_process();
  mm_nblock = vec_add_kernel.get_block_num();
  mt_nblock = sqrt_pow_kernel.get_block_num();

  // printf(
  //     "If vec_add launched first, concurrency %s; if vec_add launched second,
  //     " "concurrency %s\n", bool_to_str(can_cosched(vec_add_kernel,
  //     sqrt_pow_kernel)), bool_to_str(can_cosched(sqrt_pow_kernel,
  //     vec_add_kernel)));

  // for (unsigned i = 0; i < mm_nblock; i += mm_step) {
  //   // if (i < mm_nblock)
  vec_add_kernel.launch(KernelSliceRange{0, mm_nblock}, streams[0]);
  // }

  // for (unsigned i = 0; i < mt_nblock; i += mt_step) {
  // if (i < mt_nblock)
  sqrt_pow_kernel.launch(KernelSliceRange{0, mt_nblock}, streams[1]);
  // }
  CHECK(cudaDeviceSynchronize());
  vec_add_kernel.post_process();
  sqrt_pow_kernel.post_process();

  // not sliced
  vec_add_kernel.pre_process();
  sqrt_pow_kernel.pre_process();
  mm_nblock = vec_add_kernel.get_block_num();
  mt_nblock = sqrt_pow_kernel.get_block_num();

  start = current_seconds();

  vec_add_kernel.launch(KernelSliceRange{0, mm_nblock}, (cudaStream_t)NULL);
  sqrt_pow_kernel.launch(KernelSliceRange{0, mt_nblock}, (cudaStream_t)NULL);
  CHECK(cudaDeviceSynchronize());

  end = current_seconds();

  printf("============================== Not sliced Duration %lf\n",
         end - start);

  vec_add_kernel.post_process();
  sqrt_pow_kernel.post_process();

  // Mix
  vec_add_kernel.pre_process();
  sqrt_pow_kernel.pre_process();
  mm_nblock = vec_add_kernel.get_block_num();
  mt_nblock = sqrt_pow_kernel.get_block_num();

  start = current_seconds();

  for (unsigned i = 0, j = 0, iter = 0; i < mm_nblock || j < mt_nblock;
       iter++) {
    if (i < mm_nblock) {
      vec_add_kernel.launch(
          KernelSliceRange{i, std::min(i + mm_step, mm_nblock)},
          streams[iter % (NSTREAM / 2)]);
      i += mm_step;
    }
    for (int mt_iter = 0; mt_iter < 1; mt_iter++) {
      if (j < mt_nblock) {
        sqrt_pow_kernel.launch(
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

  vec_add_kernel.post_process();
  sqrt_pow_kernel.post_process();

  // serial
  vec_add_kernel.pre_process();
  sqrt_pow_kernel.pre_process();
  mm_nblock = vec_add_kernel.get_block_num();
  mt_nblock = sqrt_pow_kernel.get_block_num();

  start = current_seconds();

  for (unsigned i = 0, iter = 0; i < mm_nblock; i += mm_step, iter++) {
    // if (i < mm_nblock)
    vec_add_kernel.launch(KernelSliceRange{i, std::min(i + mm_step, mm_nblock)},
                          streams[iter % (NSTREAM)]);
  }

  for (unsigned i = 0, iter = 0; i < mt_nblock; i += mt_step, iter++) {
    // if (i < mt_nblock)
    sqrt_pow_kernel.launch(
        KernelSliceRange{i, std::min(i + mt_step, mt_nblock)},
        streams[iter % NSTREAM]);
  }
  CHECK(cudaDeviceSynchronize());

  end = current_seconds();

  printf("============================== Serial Duration %lf\n", end - start);

  vec_add_kernel.post_process();
  sqrt_pow_kernel.post_process();
}