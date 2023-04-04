#include "scheduler.cuh"
#include "utility.cuh"

// namespace matrix_mul
// {
__global__ void matrixMulCUDA(KernelConfig &kernel_config, KernelSlice kernel_slice);

KernelConfig pre_process();

void post_process(KernelConfig &kernel_config);

struct MatrixMulSplitArgs : KernelArgs {
  float *matrixA, *matrixB, *matrixC;
  unsigned int wA, wB;
  MatrixMulSplitArgs(float *matrixA, float *matrixB, float *matrixC, unsigned int wA,
                     unsigned int wB)
      : matrixA(matrixA), matrixB(matrixB), matrixC(matrixC), wA(wA), wB(wB) {}
};
// }  // namespace matrix_mul
