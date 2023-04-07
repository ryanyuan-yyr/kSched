/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <utility>

#include "ksched.cuh"
#include "matrix_mul.cuh"
#include "utility.cuh"

const float valB = 0.01f;

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
#define BLOCK_SIZE 32

__global__ void matrix_mul(Args args, KernelSlice kernel_slice) {
  dim3 blockIdx = kernel_slice.get_original_block_idx();

  // Block index
  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;

  // Thread index
  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;

  MatrixMulArgs *mm_args = args.as<MatrixMulArgs>();

  float *A = mm_args->matrixA, *B = mm_args->matrixB, *C = mm_args->matrixC;
  int wA = mm_args->wA, wB = mm_args->wB;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
}

void constantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

/**
 * Program main
 */
EXPORT KernelConfig pre_process() {
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  // Use a larger block size for Fermi and above
  int block_size = 32;

  dim3 dimsA(1 << 12, 1 << 12, 1);
  dim3 dimsB(1 << 12, 1 << 12, 1);

  if (dimsA.x != dimsB.y) {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }

  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x,
         dimsB.y);

  // Allocate host memory for matrices A and B
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A = (float *)malloc(mem_size_A);
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B = (float *)malloc(mem_size_B);

  // Initialize host memory
  constantInit(h_A, size_A, 1.0f);
  constantInit(h_B, size_B, valB);

  // Allocate device memory
  float *d_A, *d_B, *d_C;

  // Allocate host matrix C
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
  float *h_C = (float *)malloc(mem_size_C);

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  CHECK(cudaMalloc((void **)&d_A, mem_size_A));
  CHECK(cudaMalloc((void **)&d_B, mem_size_B));
  CHECK(cudaMalloc((void **)&d_C, mem_size_C));

  // copy host memory to device
  CHECK(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

  // Setup execution parameters
  dim3 block_dim(block_size, block_size);
  printf("threads(%d, %d)\n", block_size, block_size);
  dim3 grid_dim(dimsB.x / block_dim.x, dimsA.y / block_dim.y);
  printf("grid(%d, %d)\n", dimsB.x / block_dim.x, dimsA.y / block_dim.y);

  MatrixMulArgs args = MatrixMulArgs{d_A, d_B, d_C, dimsA.x, dimsB.x};
  MatrixMulContext context{h_A, h_B, h_C};
  return KernelConfig{(KernelPtr)matrix_mul, args, grid_dim, block_dim,
                      Context{context}};
}

DEFAULT_EXECUTE(matrix_mul);

EXPORT void post_process(Kernel &kernel) {
  MatrixMulArgs *mm_args = kernel.get_args<MatrixMulArgs>();
  dim3 dimsA(mm_args->wA, mm_args->wB, 1);
  dim3 dimsB(mm_args->wB, mm_args->wA, 1);
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);

  MatrixMulContext *context = kernel.get_context<MatrixMulContext>();
  float *h_C = context->h_C;
  float *d_C = mm_args->matrixC;
  CHECK(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

  printf("Checking computed result for correctness: \n");
  bool correct = true;

  // test relative error by the formula
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1.e-6;  // machine zero

  int errors = 0;
  for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++) {
    double abs_err = fabs(h_C[i] - (dimsA.x * valB));
    double dot_length = dimsA.x;
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;

    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i,
             h_C[i], dimsA.x * valB, eps);
      correct = false;
      errors++;
    }
    if (errors > 20) {
      printf("more than 20 errors.\n");
      break;
    }
  }
  if (correct) printf("MM PASS !\n");

  free(context->h_A);
  free(context->h_B);
  free(context->h_C);
}