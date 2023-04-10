#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <tuple>
#include <utility>

#include "ksched.cuh"

constexpr int nrepeat = 3;

__host__ int main() {
  // set up max connectioin
  const char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
  setenv(iname, "32", 1);

  Kernel vec_add_kernel{"build/vec_add.so"};
  Kernel sqrt_pow_kernel{"build/sqrt_pow.so"};

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
  Config granularity{co_kernel.get_granularity()};
  printf("Granularity %d, %d\n", granularity.first, granularity.second);
  auto subregion = std::pair<Axes<int>, Axes<int>>{
      {granularity.first * 48, granularity.second * 1},
      {granularity.first * 80, granularity.second * 4}};
  printf("Subregion (%d, %d), (%d, %d)\n", subregion.first.first,
         subregion.first.second, subregion.second.first,
         subregion.second.second);

  std::ofstream output{"data/comprehensive_tune_config"};

  for (int i = subregion.first.first; i < subregion.second.first;
       i += granularity.first) {
    for (int j = subregion.first.second; j < subregion.second.second;
         j += granularity.second) {
      Config current{i, j};
      // printf("===== Start config %d, %d\n", current.first, current.second);
      CoSchedKernels::Stat stat{};
      double current_time;
      for (Config step{128, 4}, radius{}; step.first > 1 || step.second > 1;
           radius = step * 2, step.first = step.first > 1 ? step.first / 16 : 1,
           step.second = step.second > 1 ? step.second / 2 : 1) {
        auto res =
            co_kernel.get_local_optimal(current, step * granularity, boundary,
                                        nrepeat, radius * granularity, &stat);

        current = res.first;
        current_time = res.second;
        // printf("Opt %d %d\n", current.first, current.second);
      }
      output << current_time << " ";
      // printf("config %d, %d; time %lf; steps %u, cache hit %u\n",
      // current.first,
      //        current.second, current_time, stat.steps, stat.cache_hit);
    }
    output << "\n";
  }
}