#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>

#include "ksched.cuh"

constexpr int nrepeat = 3;

__host__ int main() {
  // set up max connectioin
  const char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
  setenv(iname, "32", 1);

  Kernel vec_add_kernel{"build/vec_add.so"};
  Kernel sqrt_pow_kernel{"build/matrix_mul.so"};

  constexpr int NSTREAM = 2;
  cudaStream_t streams[NSTREAM];
  for (size_t i = 0; i < NSTREAM; i++) {
    cudaStreamCreate(streams + i);
  }

  CoSchedKernels co_kernel{vec_add_kernel, sqrt_pow_kernel, streams[0],
                           streams[1]};

  printf("Boundary %d, %d\n", co_kernel.get_boundary().first,
         co_kernel.get_boundary().second);

  // warmup
  printf("============================== Warm up Duration %lf\n",
         co_kernel.eval_cosched_time(co_kernel.get_boundary(), nrepeat));

  // Serial
  printf("============================== Not sliced Duration %lf\n",
         co_kernel.eval_cosched_time(co_kernel.get_boundary(), nrepeat));

  // Mix
  auto boundary = co_kernel.get_boundary();
  auto subregion = boundary / 16;
  Config granularity{co_kernel.get_granularity()};
  printf("Granularity %d, %d\n", granularity.first, granularity.second);

  for (auto current : {
           Config{granularity * (boundary / 2048)},
           Config{granularity * (boundary / 2048) * 2},
           Config{granularity * (boundary / 2048) * 3},
           Config{granularity * (boundary / 2048) * 4},
           Config{granularity * (boundary / 2048) * 5},
           //    Config{granularity * Config{192 / 3, 6 / 3}},
           //    Config{granularity * Config{192, 18 / 3}},
           //    Config{granularity * Config{288 / 3, 12 / 3}},
           //    Config{granularity * Config{288 / 3, 24 / 3}}
       }) {
    printf("===== Start config %d, %d\n", current.first, current.second);
    CoSchedKernels::Stat stat{};
    double current_time;
    for (Config step{boundary / 512}, radius{};
         step.first > 1 || step.second > 1;
         radius = step * 2,
         step.first = step.first > 1
                          ? step.first / std::max(boundary.first / 512 / 4, 2)
                          : 1,
         step.second =
             step.second > 1
                 ? step.second / std::max(boundary.second / 512 / 4, 2)
                 : 1) {
      auto res = co_kernel.get_local_optimal(current, step * granularity,
                                             subregion, nrepeat,
                                             radius * granularity, &stat, true);

      current = res.first;
      current_time = res.second;
      printf("Sampled opt %d %d\n", current.first, current.second);
    }
    co_kernel.flush_cache();
    printf(
        "config %d, %d; sampling time %lf, true time %lf; steps %u, cache hit "
        "%u\n",
        current.first, current.second, current_time,
        co_kernel.eval_cosched_time(current, nrepeat, false, false, false),
        stat.steps, stat.cache_hit);
  }
}