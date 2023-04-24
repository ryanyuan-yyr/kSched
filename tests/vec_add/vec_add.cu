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
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include "kosched.cuh"
#include "utility.cuh"

struct VecAddArgs {
  float *A, *B, *C;
  long n;
};

struct VecAddContext {
  float *h_A, *h_B, *h_C;
};

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
// __global__ void
// vec_add2(Args args, KernelSlice kernel_slice)
// {
// 	dim3 blockIdx = kernel_slice.get_original_block_idx();
// 	float *A = (float*)(params.getParameter(0));
// 	float *B = (float*)(params.getParameter(1));
// 	float *C = (float*)(params.getParameter(2));
// 	float *temp = (float*)(params.getParameter(3));
// 	int numElements = params.getParameter<int>(4);

// 	/****************************************************************/
// 	// rebuild blockId
// 	/****************************************************************/
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
// 	//int n = numElements;

// 	//if (threadIdx.x == 0)
// 		//printf("numElements:%d\t\t", numElements);

// 	//if (threadIdx.x == 0)
// 		//printf("i:%ld, n:%d [%ld]\t\t\t\t", i, n, n-i);
//     if (i < numElements)
//     {
// 		//if (threadIdx.x == 0)
// 			//printf("ii:%d\t", i);
//         C[i] = A[i] + B[i];

// 		for (int k =0;k < 1;k++)
// 			for (int j = 2;j < 1000;j++)
// 			{
// 				temp[j] = temp[j+1] + temp[j+2];
// 				//temp[numElements-j] = temp[numElements-j-1] +
// temp[numElements-j-2];
// 			}
//    }
//    //else if (threadIdx.x == 0)
// 		//printf("iii:%d\t", i);

// }

constexpr int unroll_len = 4;

__global__ void vec_add(Args args, KernelSlice kernel_slice) {
  dim3 blockIdx = kernel_slice.get_original_block_idx();

  VecAddArgs *va_args = args.as<VecAddArgs>();
  float *A = va_args->A;
  float *B = va_args->B;
  float *C = va_args->C;
  long numElements = va_args->n;

  long i = blockDim.x * blockIdx.x + threadIdx.x;

  for (long elem_i = i * unroll_len; elem_i < (i + 1) * unroll_len; elem_i++) {
    if (elem_i < numElements) {
      C[elem_i] = A[elem_i] + B[elem_i];
    }
  }
}

EXPORT KernelConfig pre_process() {
  long int numElements = 1 << 25;
  // long int numElements = 1 << 29;
  size_t vecAdd_size = numElements * sizeof(float);
  float *vecAdd_h_A = (float *)malloc(vecAdd_size);
  float *vecAdd_h_B =
      (float *)malloc(vecAdd_size);  // Allocate the host output vector C
  float *vecAdd_h_C = (float *)malloc(vecAdd_size);
  for (int i = 0; i < numElements; ++i) {
    vecAdd_h_A[i] = rand() / (float)RAND_MAX;
    vecAdd_h_B[i] = rand() / (float)RAND_MAX;
  }
  float *vecAdd_d_A = NULL;
  CHECK(cudaMalloc((void **)&vecAdd_d_A, vecAdd_size));
  float *vecAdd_d_B = NULL;
  CHECK(cudaMalloc((void **)&vecAdd_d_B, vecAdd_size));
  float *vecAdd_d_C = NULL;
  CHECK(cudaMalloc((void **)&vecAdd_d_C, vecAdd_size));
  CHECK(
      cudaMemcpy(vecAdd_d_A, vecAdd_h_A, vecAdd_size, cudaMemcpyHostToDevice));
  CHECK(
      cudaMemcpy(vecAdd_d_B, vecAdd_h_B, vecAdd_size, cudaMemcpyHostToDevice));

  // float *d_temp = NULL;
  // CHECK(cudaMalloc((void **)&d_temp, vecAdd_size));
  // cudaMemset(d_temp, 0, numElements);

  int vecAdd_threadsPerBlock = 128;
  int elem_per_block = vecAdd_threadsPerBlock * unroll_len;

  int vecAdd_blocksPerGrid =
      (numElements + elem_per_block - 1) / elem_per_block;
  dim3 vecAdd_blocks(vecAdd_blocksPerGrid, 1);
  dim3 vecAdd_threads(vecAdd_threadsPerBlock, 1);

  printf(
      "vectorAdd: numElements: %ld, vecAdd_blocks(%d,%d), "
      "vecAdd_threads(%d,%d)\n",
      numElements, vecAdd_blocks.x, vecAdd_blocks.y, vecAdd_threads.x,
      vecAdd_threads.y);
  VecAddArgs args = VecAddArgs{vecAdd_d_A, vecAdd_d_B, vecAdd_d_C, numElements};
  VecAddContext context{vecAdd_h_A, vecAdd_h_B, vecAdd_h_C};
  return KernelConfig{(KernelPtr)vec_add, args, vecAdd_blocks, vecAdd_threads,
                      Context{context}};
}

DEFAULT_EXECUTE(vec_add)

EXPORT void post_process(Kernel &kernel) {}