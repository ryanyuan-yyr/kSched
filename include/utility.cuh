#include <nvml.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

#include <functional>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <utility>

#ifndef UTILITY
#define UTILITY

#define CHECK(call)                                          \
  {                                                          \
    const cudaError_t error = call;                          \
    if (error != cudaSuccess) {                              \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
      fprintf(stderr, "code: %d, reason: %s\n", error,       \
              cudaGetErrorString(error));                    \
      exit(1);                                               \
    }                                                        \
  }

#define CHECK_CUBLAS(call)                                             \
  {                                                                    \
    cublasStatus_t err;                                                \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS) {                     \
      fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__, \
              __LINE__);                                               \
      exit(1);                                                         \
    }                                                                  \
  }

#define CHECK_CURAND(call)                                             \
  {                                                                    \
    curandStatus_t err;                                                \
    if ((err = (call)) != CURAND_STATUS_SUCCESS) {                     \
      fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__, \
              __LINE__);                                               \
      exit(1);                                                         \
    }                                                                  \
  }

#define CHECK_CUFFT(call)                                             \
  {                                                                   \
    cufftResult err;                                                  \
    if ((err = (call)) != CUFFT_SUCCESS) {                            \
      fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__, \
              __LINE__);                                              \
      exit(1);                                                        \
    }                                                                 \
  }

#define CHECK_CUSPARSE(call)                                               \
  {                                                                        \
    cusparseStatus_t err;                                                  \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS) {                       \
      fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__); \
      cudaError_t cuda_err = cudaGetLastError();                           \
      if (cuda_err != cudaSuccess) {                                       \
        fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                cudaGetErrorString(cuda_err));                             \
      }                                                                    \
      exit(1);                                                             \
    }                                                                      \
  }

#define EXPORT extern "C"

// fprintf(stderr, "Error: %s at %s:%d\n", (msg), __FILE__, __LINE__);
// exit(1);
#define ERR(msg) \
  { printf("Error: %s at %s:%d\n", (msg), __FILE__, __LINE__); }

inline double seconds() {
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

template <size_t SZ>
class FixSizeBox {
 private:
  char storage[SZ];

 public:
  FixSizeBox() = default;
  FixSizeBox(const FixSizeBox& other) = default;
  template <class Ty>
  FixSizeBox(Ty& data) {
    if (sizeof(Ty) > SZ) ERR("FixSizeBox: data size exceeds max size");
    *reinterpret_cast<Ty*>(storage) = data;
  }
  FixSizeBox& operator=(const FixSizeBox&) = default;
  FixSizeBox& operator=(FixSizeBox&&) = default;

  template <class Ty>
  __host__ __device__ Ty* as() {
    return reinterpret_cast<Ty*>(storage);
  }
};

template <class Ty>
class Range {
 private:
  Ty begin, end;

 public:
  Range(Ty begin, Ty end) : begin(begin), end(end) {
    if (end < begin) {
      ERR("Range: end < begin");
    }
  }
  __device__ __host__ bool contains(Ty v) { return v >= begin && v < end; }
  __device__ __host__ bool contains(Range<Ty> r) {
    return r.begin >= begin && r.end <= end;
  }
  __device__ __host__ Ty low() { return begin; }
  __device__ __host__ Ty high() { return end; }
  __device__ __host__ Ty len() { return end - begin; }
};

int get_ncore_pSM(cudaDeviceProp& devProp);

double current_seconds(void);

template <class Ty>
class Axes {
 public:
  Ty first{}, second{};

  bool operator==(const Axes& other) const {
    return first == other.first && second == other.second;
  }

  bool operator!=(const Axes& other) const { return !(*this == other); }

  Axes& operator+=(const Axes& other) {
    first += other.first;
    second += other.second;
    return *this;
  }
  Axes operator+(const Axes& other) const {
    Axes res{*this};
    return res += other;
  }

  Axes& operator-=(const Axes& other) {
    first -= other.first;
    second -= other.second;
    return *this;
  }
  Axes operator-(const Axes& other) const {
    Axes res{*this};
    return res -= other;
  }

  bool less_than(const Axes& other) const {
    return first < other.first && second < other.second;
  }
  bool less_eq(const Axes& other) const {
    return first <= other.first && second <= other.second;
  }

  Axes& operator*=(Ty coeff) {
    first *= coeff;
    second *= coeff;
    return *this;
  }

  Axes operator*(Ty coeff) const {
    Axes res{*this};
    return res *= coeff;
  }

  Axes& operator*=(const Axes& other) {
    first *= other.first;
    second *= other.second;
    return *this;
  }

  Axes operator*(const Axes& other) const {
    Axes res{*this};
    return res *= other;
  }

  Axes& operator/=(Ty coeff) {
    first /= coeff;
    second /= coeff;
    return *this;
  }

  Axes operator/(Ty coeff) const {
    Axes res{*this};
    return res /= coeff;
  }
};

template <class Ty>
class AxesHash {
 public:
  size_t operator()(const Axes<Ty>& p) const {
    auto h1 = std::hash<Ty>()(p.first);
    auto h2 = std::hash<Ty>()(p.second);
    return h1 ^ (h2 << 1);
  }
};

using Config = Axes<int>;

#endif  // UTILITY