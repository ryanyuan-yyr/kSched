struct MatrixMulArgs {
  float *matrixA, *matrixB, *matrixC;
  unsigned int wA, wB;
};

struct MatrixMulContext {
  float *h_A, *h_B, *h_C;
};
