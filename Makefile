runtest: 

TESTS := build/matrix_mul.so build/matrix_transpose.so
# TESTS := build/matrix_transpose.so
CFLAGS := -Wall -Werror -O3 -g -fPIC
NVCC_FLAGS:= -Iinclude --compiler-options="$(CFLAGS)"
NVCC := nvcc
COMMON_DEPENDENCY := include/* build/utility.o \
					include/* build/utility.o \
					include/* build/utility.o

runtests:build_runtests
	./bin/runtests

build_runtests: bin/runtests

bin/runtests: build/run_tests.o $(TESTS)
	$(NVCC) build/run_tests.o $(NVCC_FLAGS) $(TESTS) -o bin/runtests

build/run_tests.o: scripts/run_tests.cu $(COMMON_DEPENDENCY)
	$(NVCC) -c scripts/run_tests.cu $(NVCC_FLAGS) -o build/run_tests.o

build/matrix_mul.so: tests/matrix_mul/matrix_mul.cu $(COMMON_DEPENDENCY)
	$(NVCC) --shared tests/matrix_mul/matrix_mul.cu $(NVCC_FLAGS) -o build/matrix_mul.so

build/matrix_transpose.so: tests/matrix_transpose/matrix_transpose.cu $(COMMON_DEPENDENCY)
	$(NVCC) --shared tests/matrix_transpose/matrix_transpose.cu $(NVCC_FLAGS) -o build/matrix_transpose.so

build/utility.o: src/utility/utility.cu include/utility.cuh
	$(NVCC) -c src/utility/utility.cu $(NVCC_FLAGS) -o build/utility.o

clean:
	rm -r build/*