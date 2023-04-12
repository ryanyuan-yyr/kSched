#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <tuple>
#include <utility>

#include "kosched.cuh"

constexpr int nrepeat = 3;

__host__ int main() {
  // set up max connectioin
  const char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
  setenv(iname, "32", 1);

  Kernel kernel_1{"build/matrix_transpose.so"};
  Kernel kernel_2{"build/vec_add.so"};

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
  printf("============================== Warm up Duration %lf\n",
         co_kernel.eval_cosched_time(co_kernel.get_boundary(), nrepeat));

  // Serial
  printf("============================== Not sliced Duration %lf\n",
         co_kernel.eval_cosched_time(co_kernel.get_boundary(), nrepeat));

  // Mix
  Config granularity{co_kernel.get_granularity()};
  printf("Granularity %d, %d\n", granularity.first, granularity.second);
  auto subregion = std::pair<Axes<int>, Axes<int>>{
      {granularity.first * 8, granularity.second * 8},
      {granularity.first * 769, granularity.second * 385}};
  printf("Subregion (%d, %d), (%d, %d)\n", subregion.first.first,
         subregion.first.second, subregion.second.first,
         subregion.second.second);

  std::ofstream output{"data/comprehensive_tune_config"};

  for (int i = subregion.first.first; i < subregion.second.first;
       i += granularity.first) {
    for (int j = subregion.first.second; j < subregion.second.second;
         j += granularity.second) {
      output << co_kernel.eval_cosched_time({i, j}, nrepeat, false, false,
                                            false)
             << " ";
      wait_gpu_cooling(100000, 3);
    }
    output << "\n";
  }
}