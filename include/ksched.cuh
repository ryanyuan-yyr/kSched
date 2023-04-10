// CUDA runtime
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdlib.h>

#include <memory>

#include "utility.cuh"

#ifndef SCHEDULER
#define SCHEDULER

constexpr size_t MAX_ARGS_SZ = 128;
constexpr size_t MAX_CONTEXT_SZ = 128;

class KernelSlice;
class KernelConfig;
class Kernel;

typedef FixSizeBox<MAX_ARGS_SZ> Args;
typedef FixSizeBox<MAX_CONTEXT_SZ> Context;
typedef void (*KernelPtr)(Args, KernelSlice);
typedef Range<unsigned> KernelSliceRange;

using PreProcessFunc = KernelConfig (*)();
using PostProcessFunc = void (*)(Kernel&);
using ExecuteFunc = void (*)(KernelSliceRange kernel_slice_range,
                             dim3 block_dim, size_t shared_mem,
                             cudaStream_t stream, dim3 grid_dim, Args args);

/**
 * C++ name mangling
 */
const char* PREPROCESS_FUNC_NAME = "pre_process";
const char* POSTPROCESS_FUNC_NAME = "post_process";
const char* EXECUTE_FUNC_NAME = "execute";

/**
 * Functions defined by the users:
 * 1. pre_process
 * 2. execute
 * 3. post_process
 */
#define DEFAULT_EXECUTE(KERNEL_NAME)                                          \
  EXPORT                                                                      \
  void execute(KernelSliceRange kernel_slice_range, dim3 block_dim,           \
               size_t shared_mem, cudaStream_t stream, dim3 grid_dim,         \
               Args args) {                                                   \
    KERNEL_NAME<<<kernel_slice_range.len(), block_dim, shared_mem, stream>>>( \
        args, KernelSlice{grid_dim, kernel_slice_range});                     \
  }

class KernelSlice {
 private:
  dim3 grid_dim;
  KernelSliceRange kernel_slice_range;

 public:
  KernelSlice(dim3 grid_dim, KernelSliceRange kernel_slice_range)
      : grid_dim(grid_dim), kernel_slice_range(kernel_slice_range) {
    if (!KernelSliceRange{0, grid_dim.x * grid_dim.y * grid_dim.z}.contains(
            kernel_slice_range)) {
      fprintf(stderr, "kernel slice range %d %d\n", kernel_slice_range.low(),
              kernel_slice_range.high());
      ERR("Out of grid size");
    }
  }
  __device__ dim3 get_original_block_idx() {
    if (gridDim.y != 1 || gridDim.z != 1) {
      ERR("Kernel slice should be 1-dimensional");
    }
    int block_uid = blockIdx.x + kernel_slice_range.low();
    if (block_uid >= kernel_slice_range.high()) {
      ERR("out of kernel slice");
    }
    int x = block_uid % grid_dim.x;
    int z = block_uid / (grid_dim.x * grid_dim.y);
    int y = (block_uid - z * grid_dim.x * grid_dim.y) / grid_dim.x;

    return dim3(x, y, z);
  }

  __device__ dim3 get_original_grid_idx() { return grid_dim; }
};

struct KernelConfig {
  KernelPtr kernel_ptr;
  Args args;
  dim3 grid_dim;
  dim3 block_dim;
  Context context;
};

class Kernel {
 private:
  void* kernel_lib_handler;
  KernelPtr kernel_ptr;
  Args args;

  // Shape
  dim3 grid_dim;
  dim3 block_dim;

  Context context;

  // Characteristics
  // TODO

 public:
  // Kernel(const Kernel&) = default;
  // Kernel(Kernel&&) = default;

  Kernel(const char* kernel_lib_file_path) {
    kernel_lib_handler = dlopen(kernel_lib_file_path, RTLD_NOW);
    if (!kernel_lib_handler) {
      printf("Error loading kernel lib %s:%s\n", kernel_lib_file_path,
             dlerror());
    }
  }
  ~Kernel() {
    if (kernel_lib_handler) dlclose(kernel_lib_handler);
  }

  void pre_process() {
    PreProcessFunc pre_process = reinterpret_cast<PreProcessFunc>(
        dlsym(kernel_lib_handler, PREPROCESS_FUNC_NAME));
    if (!pre_process) {
      ERR("Error linking function pre_process");
      return;
    }
    KernelConfig config = pre_process();
    kernel_ptr = config.kernel_ptr, args = config.args,
    grid_dim = config.grid_dim, block_dim = config.block_dim,
    context = config.context;
  }

  void post_process() {
    PostProcessFunc post_process = reinterpret_cast<PostProcessFunc>(
        dlsym(kernel_lib_handler, POSTPROCESS_FUNC_NAME));
    if (!post_process) {
      ERR("Error linking function post_process");
    }
    post_process(*this);
  }

  template <class Arg_Ty>
  Arg_Ty* get_args() {
    return args.as<Arg_Ty>();
  }

  template <class Context_Ty>
  Context_Ty* get_context() {
    return context.as<Context_Ty>();
  }

  size_t get_block_num() const { return grid_dim.x * grid_dim.y * grid_dim.z; }

  size_t get_nthread_per_block() const {
    return block_dim.x * block_dim.y * block_dim.z;
  }

  auto __device__ __host__ get_grid_dim() const { return grid_dim; }

  void launch(KernelSliceRange kernel_slice_range) {
    kernel_ptr<<<kernel_slice_range.len(), block_dim>>>(
        args, KernelSlice{grid_dim, kernel_slice_range});
  }

  void launch(KernelSliceRange kernel_slice_range,
              size_t shared_mem_size) const {
    kernel_ptr<<<kernel_slice_range.len(), block_dim, shared_mem_size>>>(
        args, KernelSlice{grid_dim, kernel_slice_range});
  }

  void launch(KernelSliceRange kernel_slice_range, cudaStream_t stream) const {
    auto execute =
        reinterpret_cast<ExecuteFunc>(dlsym(kernel_lib_handler, "execute"));
    if (!execute) {
      ERR("Error linking function execute");
    }
    execute(kernel_slice_range, block_dim, 0, stream, grid_dim, args);
    // kernel_ptr<<<kernel_slice_range.len(), block_dim, 0, stream>>>(
    //     args, KernelSlice{grid_dim, kernel_slice_range});
  }

  void launch(KernelSliceRange kernel_slice_range, size_t shared_mem_size,
              cudaStream_t stream) const {
    kernel_ptr<<<kernel_slice_range.len(), block_dim, shared_mem_size,
                 stream>>>(args, KernelSlice{grid_dim, kernel_slice_range});
  }
};

class CoSchedKernels {
 private:
  Kernel &kernel1, &kernel2;
  cudaStream_t stream1, stream2;
  Config boundary;
  std::unordered_map<Config, double, AxesHash<int>> cosched_time_cache;

  const Axes<int> DIRECTION[4] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
  static int get_opposite_direction(int direction_idx) {
    return (direction_idx + 2) % 4;
  }

  void launch1(KernelSliceRange kernel_slice_range) {
    kernel1.launch(kernel_slice_range, stream1);
  }
  void launch2(KernelSliceRange kernel_slice_range) {
    kernel2.launch(kernel_slice_range, stream2);
  }

 public:
  CoSchedKernels(Kernel& kernel1, Kernel& kernel2, cudaStream_t stream1,
                 cudaStream_t stream2)
      : kernel1(kernel1), kernel2(kernel2), stream1(stream1), stream2(stream2) {
    kernel1.pre_process();
    kernel2.pre_process();
    boundary = {static_cast<int>(kernel1.get_block_num()),
                static_cast<int>(kernel2.get_block_num())};
  }

  ~CoSchedKernels() {
    kernel1.post_process();
    kernel2.post_process();
  }

  inline auto get_boundary() { return boundary; }

  inline auto get_granularity() {
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);
    auto block_per_sm = [&](int thread_per_block) {
      double ratio = get_ncore_pSM(dev_prop) / (double)thread_per_block;
      return ratio >= 1 ? static_cast<int>(ratio) : 1;
    };
    return Axes<int>{block_per_sm(kernel1.get_nthread_per_block()),
                     block_per_sm(kernel2.get_nthread_per_block())} *
           dev_prop.multiProcessorCount;
  }

  double eval_cosched_time(Config config, int repeat) {
    unsigned nblock_1 = kernel1.get_block_num();
    unsigned nblock_2 = kernel2.get_block_num();

    double time_sum = 0;
    for (int iter = 0; iter < repeat; iter++) {
      double start, end;
      start = current_seconds();

      for (unsigned i = 0, j = 0, iter = 0; i < nblock_1 || j < nblock_2;
           iter++) {
        if (i < nblock_1) {
          launch1(KernelSliceRange{i, std::min(i + config.first, nblock_1)});
          i += config.first;
        }
        if (j < nblock_2) {
          launch2(KernelSliceRange{j, std::min(j + config.second, nblock_2)});
          j += config.second;
        }
      }

      CHECK(cudaDeviceSynchronize());
      end = current_seconds();
      time_sum += end - start;
    }
    auto time = time_sum / repeat;
    cosched_time_cache[config] = time;
    return time;
  }

  double get_cached_eval_cosched_time(Config config, int nrepeat,
                                      bool* cached = nullptr) {
    auto obj = cosched_time_cache.find(config);
    if (obj != cosched_time_cache.end()) {
      if (cached) *cached = true;
      return obj->second;
    } else {
      if (cached) *cached = false;
      return eval_cosched_time(config, nrepeat);
    }
  }

  struct Stat {
    unsigned steps{};
    unsigned cache_hit{};
    unsigned cache_miss{};
  };

  std::pair<Config, double> get_local_optimal(Config start, Axes<int> steps,
                                              Axes<int> boundary, int nrepeat,
                                              Axes<int> radius = {},
                                              Stat* stat = nullptr) {
    int untested_neighbor_bit_mask = 0b1111;
    Config current = start;
    bool cache_stat;
    double current_time =
        get_cached_eval_cosched_time(current, nrepeat, &cache_stat);
    if (stat) {
      if (cache_stat)
        stat->cache_hit++;
      else
        stat->cache_miss++;
    }
    while (true) {
      int min_direct = -1;
      double min_time = std::numeric_limits<double>::max();
      for (size_t i = 0; i < 4; i++) {
        if (!(untested_neighbor_bit_mask & (1 << i))) {
          continue;
        }
        auto neighbor = current + DIRECTION[i] * steps;

        // is in the space
        bool in_space = true;
        if (!Config{}.less_than(neighbor) || !neighbor.less_eq(boundary))
          in_space = false;
        if (radius != Config{}) {
          if (!neighbor.less_eq(start + radius)) in_space = false;
          if (neighbor.less_than(start +
                                 Axes<int>{-radius.first, radius.second}))
            in_space = false;
          if (neighbor.less_than(start +
                                 Axes<int>{radius.first, -radius.second}))
            in_space = false;
        }
        if (!in_space) {
          // printf("Not in space %d %d\n", neighbor.first, neighbor.second);
          continue;
        }

        double neighbor_time =
            get_cached_eval_cosched_time(neighbor, nrepeat, &cache_stat);
        if (stat) {
          if (cache_stat)
            stat->cache_hit++;
          else
            stat->cache_miss++;
        }
        if (neighbor_time < min_time) {
          min_direct = i;
          min_time = neighbor_time;
        }
      }
      if (min_direct == -1) {
        // printf("Current %d %d, neighbor %d %d")
        throw std::logic_error("No neighbor in range");
      }
      if (min_time >= current_time)
        return {current, current_time};
      else {
        current += DIRECTION[min_direct] * steps;
        current_time = min_time;
        untested_neighbor_bit_mask =
            0b1111 & (~(1 << get_opposite_direction(min_direct)));
        if (stat) {
          stat->steps++;
        }
      }
    }
  }
};

#endif