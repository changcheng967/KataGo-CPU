// DCU fp16 GEMM benchmark: hipblasGemmEx with fp16 inputs, fp32 compute.
// This is the precision the card is fastest at (packed half2 datapath on gfx906).
// Build: hipcc -O3 -std=c++14 gpubench16.cpp -lhipblas -o gpubench16
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas.h>

#define CHECK_HIP(x) do { hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %d at %d\n",e,__LINE__);exit(1);} } while(0)
#define CHECK_HB(x) do { hipblasStatus_t s=(x); if(s!=HIPBLAS_STATUS_SUCCESS){printf("hipBLAS err %d at %d\n",(int)s,__LINE__);exit(1);} } while(0)

int main() {
  hipblasHandle_t h; CHECK_HB(hipblasCreate(&h));
  for(int N : {2048, 4096}) {
    size_t sz = (size_t)N * N;
    __half *dA, *dB; float *dC;
    CHECK_HIP(hipMalloc(&dA, sz * 2)); CHECK_HIP(hipMalloc(&dB, sz * 2)); CHECK_HIP(hipMalloc(&dC, sz * 4));
    CHECK_HIP(hipMemset(dA, 0, sz * 2)); CHECK_HIP(hipMemset(dB, 0, sz * 2));
    hipblasHalf ha = (hipblasHalf)0x3C00, hb = (hipblasHalf)0x0000;  // 1.0f, 0.0f in fp16 bits
    // warmup: fp16 in/out (old DTK API; accumulate precision is the card's choice)
    CHECK_HB(hipblasHgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N,
                          &ha, (hipblasHalf*)dA, N, (hipblasHalf*)dB, N, &hb, (hipblasHalf*)dC, N));
    CHECK_HIP(hipDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    for(int r = 0; r < 20; r++)
      CHECK_HB(hipblasHgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, N, N, N,
                            &ha, (hipblasHalf*)dA, N, (hipblasHalf*)dB, N, &hb, (hipblasHalf*)dC, N));
    CHECK_HIP(hipDeviceSynchronize());
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20;
    printf("fp16 GEMM %d^3: %.2f ms -> %.1f TFLOP/s\n", N, s * 1e3, 2.0 * (double)N * N * N / s / 1e12);
    CHECK_HIP(hipFree(dA)); CHECK_HIP(hipFree(dB)); CHECK_HIP(hipFree(dC));
  }
  printf("GPUBENCH16_DONE\n");
  return 0;
}
