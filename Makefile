runtest: 

TESTS := build/matrix_mul.so build/matrix_transpose.so build/sqrt_pow.so
# TESTS := build/matrix_transpose.so
CFLAGS := -Wall -Werror -O3 -g -fPIC
NVCC_FLAGS:= -Iinclude --compiler-options="$(CFLAGS)"
NVCC := nvcc
COMMON_DEPENDENCY := include/* build/utility.o

runtests:build_runtests
	./bin/runtests

build_runtests: bin/runtests

bin/runtests: build/run_tests.o $(TESTS) $(COMMON_DEPENDENCY)
	$(NVCC) build/run_tests.o $(NVCC_FLAGS) $(TESTS) build/utility.o -o bin/runtests

build/run_tests.o: scripts/run_tests.cu $(COMMON_DEPENDENCY)
	$(NVCC) -c scripts/run_tests.cu $(NVCC_FLAGS) -o build/run_tests.o

build/matrix_mul.so: tests/matrix_mul/matrix_mul.cu $(COMMON_DEPENDENCY)
	$(NVCC) --shared tests/matrix_mul/matrix_mul.cu $(NVCC_FLAGS) -o build/matrix_mul.so

build/matrix_transpose.so: tests/matrix_transpose/matrix_transpose.cu $(COMMON_DEPENDENCY)
	$(NVCC) --shared tests/matrix_transpose/matrix_transpose.cu $(NVCC_FLAGS) -o build/matrix_transpose.so

build/sqrt_pow.so: tests/sqrt_pow/sqrt_pow.cu $(COMMON_DEPENDENCY)
	$(NVCC) --shared tests/sqrt_pow/sqrt_pow.cu $(NVCC_FLAGS) -o build/sqrt_pow.so

build/utility.o: src/utility/utility.cu include/utility.cuh
	$(NVCC) -c src/utility/utility.cu $(NVCC_FLAGS) -o build/utility.o

clean:
	rm -r build/*