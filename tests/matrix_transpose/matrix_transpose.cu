#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <utility>

#include "ksched.cuh"
#include "matrix_transpose.cuh"
#include "utility.cuh"

// Each block transposes/copies a tile of TILE_DIM x TILE_DIM elements
// using TILE_DIM x BLOCK_ROWS threads, so that each thread transposes
// TILE_DIM/BLOCK_ROWS elements.  TILE_DIM must be an integral multiple of
// BLOCK_ROWS

#define TILE_DIM 16
#define BLOCK_ROWS 16

// This sample assumes that MATRIX_SIZE_X = MATRIX_SIZE_Y
int MATRIX_SIZE_X = 1024;
int MATRIX_SIZE_Y = 1024;
int MUL_FACTOR = TILE_DIM;

#define FLOOR(a, b) (a - (a % b))

// Compute the tile size necessary to illustrate performance cases for SM20+
// hardware
int MAX_TILES = (FLOOR(MATRIX_SIZE_X, 512) * FLOOR(MATRIX_SIZE_Y, 512)) /
                (TILE_DIM * TILE_DIM);

// Number of repetitions used for timing.  Two sets of repetitions are
// performed: 1) over kernel launches and 2) inside the kernel over just the
// loads and stores

#define NUM_REPS 1

// -------------------------------------------------------
// Transposes
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

__global__ void matrix_transpose(Args args, KernelSlice kernel_slice) {
  MatrixTransposeArgs *mt_args = args.as<MatrixTransposeArgs>();
  float *idata = mt_args->idata;
  float *odata = mt_args->odata;
  int width = mt_args->width;
  int height = width;

  /****************************************************************/
  // rebuild blockId
  dim3 blockIdx = kernel_slice.get_original_block_idx();
  /****************************************************************/
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

  int index_in = xIndex + width * yIndex;
  int index_out = yIndex + height * xIndex;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    odata[index_out + i] = idata[index_in + i * width];
  }
}

const size_t trnsp_size = TILE_DIM * (1 << 12);
const size_t trnsp_mem_size =
    static_cast<size_t>(sizeof(float) * trnsp_size * trnsp_size);

EXPORT KernelConfig pre_process() {
  if (trnsp_size % TILE_DIM != 0) {
    printf(
        "Matrix size must be integral multiple of tile size\nExiting...\n\n");
    exit(EXIT_FAILURE);
  }
  dim3 trnsp_grid(trnsp_size / TILE_DIM, trnsp_size / TILE_DIM),
      trnsp_threads(TILE_DIM, BLOCK_ROWS);

  if (trnsp_grid.x < 1 || trnsp_grid.y < 1) {
    printf("trnsp_grid size computation incorrect in test \nExiting...\n\n");
    exit(EXIT_FAILURE);
  }

  float *h_idata = (float *)malloc(trnsp_mem_size);
  float *h_odata = (float *)malloc(trnsp_mem_size);

  float *d_idata, *d_odata;
  CHECK(cudaMalloc((void **)&d_idata, trnsp_mem_size));
  CHECK(cudaMalloc((void **)&d_odata, trnsp_mem_size));

  for (size_t i = 0; i < (trnsp_size * trnsp_size); ++i) h_idata[i] = (float)i;

  CHECK(cudaMemcpy(d_idata, h_idata, trnsp_mem_size, cudaMemcpyHostToDevice));
  printf(
      "\nMatrix size: %lux%lu (%dx%d tiles), tile size: %dx%d, block size: "
      "%dx%d\n\n",
      trnsp_size, trnsp_size, trnsp_grid.x, trnsp_grid.y, TILE_DIM, TILE_DIM,
      trnsp_threads.x, trnsp_threads.y);

  MatrixTransposeArgs args{d_idata, d_odata, trnsp_size};
  MatrixTransposeContext context{h_idata, h_odata};
  return KernelConfig{(KernelPtr)matrix_transpose, Args{args}, trnsp_grid,
                      trnsp_threads, Context{context}};
}

DEFAULT_EXECUTE(matrix_transpose)

EXPORT void post_process(Kernel &kernel) {
  MatrixTransposeArgs *args = kernel.get_args<MatrixTransposeArgs>();
  MatrixTransposeContext *context =
      kernel.get_context<MatrixTransposeContext>();
  auto width = args->width;
  auto h_odata = context->h_odata;
  CHECK(
      cudaMemcpy(h_odata, args->odata, trnsp_mem_size, cudaMemcpyDeviceToHost));
  bool correct = true;
  for (size_t i = 0; i < width; i++) {
    for (size_t j = 0; j < width; j++) {
      float true_value = i + j * width;
      float real_value = h_odata[i * width + j];
      if (true_value != real_value) {
        printf("Matrix_transpose error: i %lu, j %lu, expected %f, found %f\n", i,
               j, true_value, real_value);
        correct = false;
        goto outer;
      }
    }
  }
outer:

  printf(correct ? "Matrix transpose correct\n" : "Matrix transpose error\n");

  CHECK(cudaFree(args->idata));
  CHECK(cudaFree(args->odata));
  free(context->h_idata);
  free(context->h_odata);
}