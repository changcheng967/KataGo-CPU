// DCU compute smoke benchmark: HBM bandwidth + rocblas fp32 SGEMM TFLOPs.
// Build: hipcc -O3 -std=c++14 gpubench.cpp -lrocblas -o gpubench
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <rocblas.h>

#define CHECK_HIP(x) do { hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %d at %d\n",e,__LINE__);exit(1);} } while(0)
#define CHECK_RB(x) do { rocblas_status s=(x); if(s!=rocblas_status_success){printf("rocBLAS err %d at %d\n",(int)s,__LINE__);exit(1);} } while(0)

__global__ void copyKernel(const float4* src, float4* dst, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if(i < n) dst[i] = src[i];
}

int main() {
  int nd; CHECK_HIP(hipGetDeviceCount(&nd));
  printf("devices: %d\n", nd);
  hipDeviceProp_t p; CHECK_HIP(hipGetDeviceProperties(&p, 0));
  printf("gpu0: %s | SM %d.%d | CU %d | mem %.1f GB\n",
         p.name, p.major, p.minor, p.multiProcessorCount,
         p.totalGlobalMem / 1e9);

  // HBM bandwidth: 1 GiB read + write per pass
  size_t bytes = (size_t)1 << 30;
  size_t n4 = bytes / 16;
  float4 *a, *b;
  CHECK_HIP(hipMalloc(&a, bytes)); CHECK_HIP(hipMalloc(&b, bytes));
  CHECK_HIP(hipMemset(a, 1, bytes));
  int block = 256, grid = (int)((n4 + block - 1) / block);
  copyKernel<<<grid, block>>>((float4*)a, (float4*)b, n4);
  CHECK_HIP(hipDeviceSynchronize());
  auto t0 = std::chrono::steady_clock::now();
  for(int r = 0; r < 10; r++) copyKernel<<<grid, block>>>((float4*)a, (float4*)b, n4);
  CHECK_HIP(hipDeviceSynchronize());
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 10;
  printf("HBM bandwidth: %.0f GB/s (r+w of 2GiB per pass)\n", 2.0 * bytes / s / 1e9);
  CHECK_HIP(hipFree(a)); CHECK_HIP(hipFree(b));

  // rocblas fp32 GEMM
  rocblas_handle h; CHECK_RB(rocblas_create_handle(&h));
  const int N = 4096;
  size_t sz = (size_t)N * N;
  float *dA, *dB, *dC;
  CHECK_HIP(hipMalloc(&dA, sz * 4)); CHECK_HIP(hipMalloc(&dB, sz * 4)); CHECK_HIP(hipMalloc(&dC, sz * 4));
  CHECK_HIP(hipMemset(dA, 0, sz * 4)); CHECK_HIP(hipMemset(dB, 0, sz * 4));
  float alpha = 1.f, beta = 0.f;
  CHECK_RB(rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none, N, N, N, &alpha, dA, N, dB, N, &beta, dC, N));
  CHECK_HIP(hipDeviceSynchronize());
  t0 = std::chrono::steady_clock::now();
  for(int r = 0; r < 20; r++)
    CHECK_RB(rocblas_sgemm(h, rocblas_operation_none, rocblas_operation_none, N, N, N, &alpha, dA, N, dB, N, &beta, dC, N));
  CHECK_HIP(hipDeviceSynchronize());
  s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20;
  printf("fp32 GEMM 4096^3: %.2f ms -> %.1f TFLOP/s\n", s * 1e3, 2.0 * (double)N * N * N / s / 1e12);
  printf("GPUBENCH_DONE\n");
  return 0;
}
