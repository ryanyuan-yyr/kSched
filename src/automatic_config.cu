// #define SHOW_TRACE_EXP

#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>

#include "kosched.cuh"

constexpr int nrepeat = 3;

__host__ int main() {
  // set up max connectioin
  const char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
  setenv(iname, "32", 1);

  const char* tests[] = {
   /*0*/ "build/sqrt_pow.so", 
   /*1*/ "build/matrix_mul.so", 
   /*2*/ "build/matrix_transpose.so", 
   /*3*/ "build/vec_add.so"
  };

  #ifdef SHOW_TRACE_EXP
  Kernel kernel_1{tests[3]};
  Kernel kernel_2{tests[1]};
  #else
  Kernel kernel_1{tests[0]};
  Kernel kernel_2{tests[1]};
  #endif
  bool sampling = true;

  constexpr int NSTREAM = 2;
  cudaStream_t streams[NSTREAM];
  for (size_t i = 0; i < NSTREAM; i++) {
    cudaStreamCreate(streams + i);
  }

  CoSchedKernels co_kernel{kernel_1, kernel_2, streams[0],
                           streams[1]};

  printf("Boundary %d, %d\n", co_kernel.get_boundary().first,
         co_kernel.get_boundary().second);

  // warmup
  // printf("============================== Warm up Duration %lf\n",
  //        co_kernel.eval_cosched_time(co_kernel.get_boundary(), nrepeat));

  // Serial
  printf("============================== Not sliced Duration %lf\n",
         co_kernel.eval_cosched_time(co_kernel.get_boundary(), nrepeat, false, false, false, nullptr, 128));

  // Mix
  Config granularity{co_kernel.get_granularity()};
  auto boundary = co_kernel.get_boundary();
  #ifdef SHOW_TRACE_EXP
  auto subregion = granularity*Config{60, 60};
  #else
  auto subregion = boundary; // / 16;
  #endif
  printf("Granularity %d, %d\n", granularity.first, granularity.second);

  printf("Config %d, %d\n", boundary.first / granularity.first, boundary.second / granularity.second);

  double sampling_time = 0;
  constexpr int min_step = 1;

  #ifdef SHOW_TRACE_EXP
  for (auto current : {
           Config{granularity * Config{9, 44}}
       })
  #else
  for (auto current : {
           Config{granularity * (boundary / 2048)},
           Config{granularity * (boundary / 2048) * 2},
           Config{granularity * (boundary / 2048) * 3},
           Config{granularity * (boundary / 2048) * 4},
           Config{granularity * (boundary / 2048) * 5},
       })
  #endif
  {
    printf("===== Start config %d, %d\n", current.first, current.second);
    auto sampling_start_time = current_seconds();
    CoSchedKernels::Stat stat{};
    double current_time;
    #ifdef SHOW_TRACE_EXP
    Config step{20, 20};
    auto next_step = [&](Config prev_step){return Config{
      // step.first > min_step? step.first / 2 : 1, 
      // step.second > min_step? step.second / 2 : 1
      step / 2
    };};
    #else
    Config step{boundary / 512};
    auto next_step = [&](Config prev_step){return Config{
      // step.first > min_step
      //                 ? step.first / std::max(boundary.first / 512 / 4, 2)
      //                 : min_step, 
      // step.second > min_step
      //         ? step.second / std::max(boundary.second / 512 / 4, 2)
      //         : min_step
      step.first / std::max(boundary.first / 512 / 4, 2), 
      step.second / std::max(boundary.second / 512 / 4, 2)
    };};
    #endif
    for (
      Config radius{};
      step.first >= min_step || step.second >= min_step;
      // radius = step * 2,
      step = next_step(step)) {
      auto res = co_kernel.get_local_optimal(
          current, step * granularity, subregion, 
          #if SHOW_TRACE_EXP
          128,
          #else
          sampling ? 1 : nrepeat,
          #endif
          radius * granularity, &stat, sampling);

      current = res.first;
      current_time = res.second;
      printf("Sampled opt config %d %d (step %d %d)\n", current.first / granularity.first, current.second / granularity.second, step.first, step.second);
    }
    auto sampling_end_time = current_seconds();
    sampling_time += sampling_end_time - sampling_start_time;
    // co_kernel.flush_cache();
    printf(
        "config %d, %d; sampling time %lf, true time %lf; steps %u, cache hit "
        "%u\n",
        current.first, current.second, current_time,
        co_kernel.eval_cosched_time(current, nrepeat, false, false, false, nullptr, 32),
        stat.steps, stat.cache_hit);
  }

  printf("Time taken to do searching: %lf\n", sampling_time);
}