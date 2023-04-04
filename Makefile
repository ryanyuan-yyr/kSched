runtest: 

TESTS=build/matrix_mul.o
NVCC_FLAGS= -Iinclude
NVCC = nvcc

runtests:build_runtests
	./bin/runtests

build_runtests: bin/runtests

bin/runtests: build/run_tests.o $(TESTS)
	$(NVCC) build/run_tests.o $(NVCC_FLAGS) $(TESTS) -o bin/runtests

build/run_tests.o: scripts/run_tests.cu
	$(NVCC) -c -O3 scripts/run_tests.cu $(NVCC_FLAGS) -o build/run_tests.o

build/matrix_mul.o: tests/matrix_mul/matrix_mul_split.cu
	$(NVCC) -c -O3 tests/matrix_mul/matrix_mul_split.cu $(NVCC_FLAGS) -o build/matrix_mul.o

clean:
	rm -r build/*