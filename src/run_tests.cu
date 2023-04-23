#include <cuda_runtime.h>

#include <algorithm>

#include "kosched.cuh"

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

  const char* tests[] = {
     /*0*/ "build/sqrt_pow.so", 
     /*1*/ "build/matrix_mul.so", 
     /*2*/ "build/matrix_transpose.so", 
     /*3*/ "build/vec_add.so"
    };
  
    Kernel kernel_1{tests[3]};
    Kernel kernel_2{tests[1]};

  constexpr int NSTREAM = 2;
  cudaStream_t streams[NSTREAM];
  for (size_t i = 0; i < NSTREAM; i++) {
    cudaStreamCreate(streams + i);
  }


  CoSchedKernels co_kernel{kernel_1, kernel_2, streams[0],
                          streams[1]};

  // Kernel unrelated{"build/matrix_transpose.so"};
  // unrelated.pre_process();


  // Motivation test config
  // constexpr int mm_step = 3 * 1 * 128 * 8, mt_step = 3 * 4 * 4 * 1;

  // warmup
  // for(size_t i = 0; i < 32; i++)
  //   co_kernel.eval_cosched_time(co_kernel.get_granularity(), 1, false, false, false);

  // Serial
  // for(size_t i = 0; i < 256; i++){
  //   printf("============================== Not sliced Duration %lf\n",
  //         co_kernel.eval_cosched_time(co_kernel.get_boundary(), 1, false, false, false));
  // }

  // Mix
  // for(size_t i = 0; i < 240; i++)
  //   printf("============================== Mix Duration %lf\n", co_kernel.eval_cosched_time(co_kernel.get_granularity() * Config{9, 3}, 5, false, false,
  //                                             false));
  for(size_t i = 0; i < 256; i++)
  printf("============================== Mix Duration %lf\n", co_kernel.eval_cosched_time(co_kernel.get_granularity() * Config{1, 32}, 1, false, false,
                                              false));

  // printf("============================== Mix Duration %lf\n", co_kernel.eval_cosched_time(co_kernel.get_granularity() * Config{1, 2}, 1, false, false,
  //                                             false));

  // printf("============================== Mix Duration %lf\n", co_kernel.eval_cosched_time(co_kernel.get_granularity() * Config{5, 2}, 5, false, false,
  //                                             false));

  // unrelated.post_process();
}