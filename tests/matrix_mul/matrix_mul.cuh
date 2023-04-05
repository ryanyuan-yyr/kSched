struct MatrixMulArgs {
  float *matrixA, *matrixB, *matrixC;
  unsigned int wA, wB;
  MatrixMulArgs(float *matrixA, float *matrixB, float *matrixC, unsigned int wA,
                unsigned int wB)
      : matrixA(matrixA), matrixB(matrixB), matrixC(matrixC), wA(wA), wB(wB) {}
};

struct MatrixMulContext {
  float *h_A, *h_B, *h_C;
};
