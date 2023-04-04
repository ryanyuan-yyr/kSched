#include <cuda_runtime.h>

#include "../tests/matrix_mul/matrix_mul_split.cuh"

__host__ int main() {
  auto kernel_config = pre_process();
  for (size_t i = 0; i < kernel_config.get_block_num(); i++) {
    kernel_config.launch(i);
  }

  post_process(kernel_config);
}