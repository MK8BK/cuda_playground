// simple matrix addition profiling with nvidia nsight compute
//
//
//

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <iostream>
#include <nvtx3/nvtx3.hpp>

using numT = float;

// comes from the intro to cuda c++ in the cuda programming guide
#define CUDA_CHECK(expr_to_check)                                              \
  do {                                                                         \
    cudaError_t result = expr_to_check;                                        \
    if (result != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA Runtime Error: %s:%i:%d = %s\n", __FILE__,         \
              __LINE__, result, cudaGetErrorString(result));                   \
    }                                                                          \
  } while (0)

const int32_t MATRIX_SIDE{1000};
const int32_t MATRIX_TOTAL_SIZE{MATRIX_SIDE * MATRIX_SIDE};
const int32_t MATRIX_TOTAL_SIZE_BYTE{MATRIX_TOTAL_SIZE * sizeof(numT)};
const int32_t MAX_MOD{static_cast<int32_t>(1e5)};
const numT EPS_NUM_T{static_cast<numT>(1e-5)};

inline numT modBound(numT x) {
  return x - static_cast<int32_t>(x / MAX_MOD) * MAX_MOD;
}

void initRandom(numT *arr, int32_t n) {
  for (int32_t i{}; i < n; ++i)
    arr[i] = modBound(static_cast<numT>(std::rand()));
}

__global__ void additionKernel(const numT *A, const numT *B, numT *C,
                               const int32_t n) {
  // adds A, B into C, assumes n elements within these arrays
  // used with 1d grid and 1d block size
  int32_t i{static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x)};
  if (i < n)
    C[i] = A[i] + B[i];
}

//#define CUDA_MALLOC_HOST

int main() {
  nvtxRangePush("Starting on cpu");

  numT *h_A, *h_B, *h_C, *h_C2;
  numT *d_A, *d_B, *d_C;

  nvtxRangePush("Allocating on cpu");
#ifdef CUDA_MALLOC_HOST
  CUDA_CHECK(cudaMallocHost(&h_A, MATRIX_TOTAL_SIZE_BYTE));
#else
  h_A = new numT[MATRIX_TOTAL_SIZE];
#endif // CUDA_MALLOC_HOST
  if (!h_A)
    return EXIT_FAILURE;
#ifdef CUDA_MALLOC_HOST
  CUDA_CHECK(cudaMallocHost(&h_B, MATRIX_TOTAL_SIZE_BYTE));
#else
  h_B = new numT[MATRIX_TOTAL_SIZE];
#endif // CUDA_MALLOC_HOST
  if (!h_B)
    return EXIT_FAILURE;
#ifdef CUDA_MALLOC_HOST
  CUDA_CHECK(cudaMallocHost(&h_C, MATRIX_TOTAL_SIZE_BYTE));
#else
  h_C = new numT[MATRIX_TOTAL_SIZE];
#endif // CUDA_MALLOC_HOST
  if (!h_C)
    return EXIT_FAILURE;
  h_C2 = new numT[MATRIX_TOTAL_SIZE];
  if (!h_C2)
    return EXIT_FAILURE;
  nvtxRangePop(); // ending allocation on cpu

  nvtxRangePush("Initializing on cpu");
  initRandom(h_A, MATRIX_TOTAL_SIZE);
  initRandom(h_B, MATRIX_TOTAL_SIZE);
  nvtxRangePop(); // ending initializing on cpu

  nvtxRangePush("Allocating on gpu");
  CUDA_CHECK(cudaMalloc(&d_A, MATRIX_TOTAL_SIZE_BYTE));
  CUDA_CHECK(cudaMalloc(&d_B, MATRIX_TOTAL_SIZE_BYTE));
  CUDA_CHECK(cudaMalloc(&d_C, MATRIX_TOTAL_SIZE_BYTE));
  nvtxRangePop(); // ending allocation on gpu

  nvtxRangePush("Starting copy host to device");
  CUDA_CHECK(
      cudaMemcpy(d_A, h_A, MATRIX_TOTAL_SIZE_BYTE, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_B, h_B, MATRIX_TOTAL_SIZE_BYTE, cudaMemcpyHostToDevice));
  nvtxRangePop(); // ending copy host to device

  nvtxRangePush("Starting kernel gpu addition");
  int32_t blockSize{256};
  dim3 bD(blockSize);
  dim3 gD((blockSize + MATRIX_TOTAL_SIZE - 1) / blockSize);
  additionKernel<<<gD, bD>>>(d_A, d_B, d_C, MATRIX_TOTAL_SIZE);
  CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop(); // kernel done

  nvtxRangePush("Starting copy device to host");
  CUDA_CHECK(
      cudaMemcpy(h_C, d_C, MATRIX_TOTAL_SIZE_BYTE, cudaMemcpyDeviceToHost));
  nvtxRangePop(); // ending copy device to host

  nvtxRangePush("Starting cpu addition");
  for (int32_t i{}; i < MATRIX_TOTAL_SIZE; i++)
    h_C2[i] = h_A[i] + h_B[i];
  nvtxRangePop(); // ending cpu addition

  nvtxRangePush("Starting verification");
  for (int32_t i{}; i < MATRIX_TOTAL_SIZE; i++)
    if (static_cast<numT>(abs(h_C[i] - h_C2[i]) > EPS_NUM_T)) {
      std::cerr << "Errors in calculations\n";
      return EXIT_FAILURE;
    }
  nvtxRangePop(); // ending verification

  nvtxRangePush("Cleaning up");
#ifdef CUDA_MALLOC_HOST
  CUDA_CHECK(cudaFree(h_A));
  CUDA_CHECK(cudaFree(h_B));
  CUDA_CHECK(cudaFree(h_C));
#else
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
#endif // CUDA_MALLOC_HOST
  // always on host
  delete[] h_C2;

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  nvtxRangePop();

  nvtxRangePop(); // ending on cpu
  return EXIT_SUCCESS;
}
