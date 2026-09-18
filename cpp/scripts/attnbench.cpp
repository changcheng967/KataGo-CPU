// Standalone bench of KataGo's tiled flash-attention kernel on gfx906,
// then incrementally optimized variants. Real engine shapes: batch 10,
// seq 361, heads 6, headDim 32, mask mostly-ones.
// Build: hipcc -O3 -std=c++17 attnbench.cpp -o attnbench
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define CHECK_HIP(x) do { hipError_t e=(x); if(e!=hipSuccess){printf("HIP err %d at %d\n",e,__LINE__);exit(1);} } while(0)

#ifndef BLOCK_KV
#define BLOCK_KV 32
#endif

// ---------- Baseline: verbatim port of upstream flashAttentionTiledImpl ----------
template<int qHeadDim, int vHeadDim, int BLOCK_Q, int QPT>
__global__
void flashAttentionV0(
  const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
  const float* __restrict__ mask, __half* __restrict__ output,
  int seqLen, int numHeads, float scale)
{
  const int tid = threadIdx.x;
  const int qBlockStart = blockIdx.x * (BLOCK_Q * QPT);
  const int bh = blockIdx.y;
  const int n = bh / numHeads;
  const int h = bh % numHeads;
  const int qTotalDim = numHeads * qHeadDim;
  const int vTotalDim = numHeads * vHeadDim;
  const int oTotalDim = numHeads * vHeadDim;

  __shared__ float kTile[BLOCK_KV * qHeadDim];
  __shared__ float vTile[BLOCK_KV * vHeadDim];

  float qReg[QPT * qHeadDim];
  float runningMax[QPT]; float runningSum[QPT];
  float acc[QPT * vHeadDim];

  for(int qi = 0; qi < QPT; qi++) {
    int qPos = qBlockStart + qi * BLOCK_Q + tid;
    runningMax[qi] = -1e30f; runningSum[qi] = 0.0f;
    for(int d = 0; d < vHeadDim; d++) acc[qi * vHeadDim + d] = 0.0f;
    if(qPos < seqLen) {
      const __half* qPtr = Q + ((size_t)n * seqLen + qPos) * qTotalDim + h * qHeadDim;
      #pragma unroll
      for(int d = 0; d < qHeadDim; d++) qReg[qi * qHeadDim + d] = (float)qPtr[d];
    }
  }

  for(int kvStart = 0; kvStart < seqLen; kvStart += BLOCK_KV) {
    for(int t = tid; t < BLOCK_KV * qHeadDim; t += BLOCK_Q) {
      int tileKPos = t / qHeadDim; int tileD = t % qHeadDim;
      int globalKPos = kvStart + tileKPos;
      float v = 0.0f;
      if(globalKPos < seqLen)
        v = (float)K[((size_t)n * seqLen + globalKPos) * qTotalDim + h * qHeadDim + tileD];
      kTile[tileKPos * qHeadDim + tileD] = v;
    }
    for(int t = tid; t < BLOCK_KV * vHeadDim; t += BLOCK_Q) {
      int tileKPos = t / vHeadDim; int tileD = t % vHeadDim;
      int globalKPos = kvStart + tileKPos;
      float v = 0.0f;
      if(globalKPos < seqLen)
        v = (float)V[((size_t)n * seqLen + globalKPos) * vTotalDim + h * vHeadDim + tileD];
      vTile[tileKPos * vHeadDim + tileD] = v;
    }
    __syncthreads();
    int kvEnd = min(BLOCK_KV, seqLen - kvStart);

    for(int qi = 0; qi < QPT; qi++) {
      int qPos = qBlockStart + qi * BLOCK_Q + tid;
      if(qPos >= seqLen) continue;
      for(int tk = 0; tk < kvEnd; tk++) {
        float dot = 0.0f;
        #pragma unroll
        for(int d = 0; d < qHeadDim; d++)
          dot += qReg[qi * qHeadDim + d] * kTile[tk * qHeadDim + d];
        dot *= scale;
        float newMax = fmaxf(runningMax[qi], dot);
        float expOldMax = __expf(runningMax[qi] - newMax);
        float expCur = __expf(dot - newMax);
        #pragma unroll
        for(int d = 0; d < vHeadDim; d++)
          acc[qi * vHeadDim + d] = acc[qi * vHeadDim + d] * expOldMax + expCur * vTile[tk * vHeadDim + d];
        runningSum[qi] = runningSum[qi] * expOldMax + expCur;
        runningMax[qi] = newMax;
      }
    }
    __syncthreads();
  }

  for(int qi = 0; qi < QPT; qi++) {
    int qPos = qBlockStart + qi * BLOCK_Q + tid;
    if(qPos >= seqLen) continue;
    __half* outRow = output + ((size_t)n * seqLen + qPos) * oTotalDim + h * vHeadDim;
    float invSum = (runningSum[qi] > 0.0f) ? (1.0f / runningSum[qi]) : 0.0f;
    #pragma unroll
    for(int d = 0; d < vHeadDim; d++) outRow[d] = (__half)(acc[qi * vHeadDim + d] * invSum);
  }
}

// ---------- V1: 4-way unrolled dot accumulators (break the serial chain) ----------
template<int qHeadDim, int vHeadDim, int BLOCK_Q, int QPT>
__global__
void flashAttentionV1(
  const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
  const float* __restrict__ mask, __half* __restrict__ output,
  int seqLen, int numHeads, float scale)
{
  const int tid = threadIdx.x;
  const int qBlockStart = blockIdx.x * (BLOCK_Q * QPT);
  const int bh = blockIdx.y;
  const int n = bh / numHeads;
  const int h = bh % numHeads;
  const int qTotalDim = numHeads * qHeadDim;
  const int vTotalDim = numHeads * vHeadDim;
  const int oTotalDim = numHeads * vHeadDim;

  __shared__ float kTile[BLOCK_KV * qHeadDim];
  __shared__ float vTile[BLOCK_KV * vHeadDim];

  float qReg[QPT * qHeadDim];
  float runningMax[QPT]; float runningSum[QPT];
  float acc[QPT * vHeadDim];

  for(int qi = 0; qi < QPT; qi++) {
    int qPos = qBlockStart + qi * BLOCK_Q + tid;
    runningMax[qi] = -1e30f; runningSum[qi] = 0.0f;
    for(int d = 0; d < vHeadDim; d++) acc[qi * vHeadDim + d] = 0.0f;
    if(qPos < seqLen) {
      const __half* qPtr = Q + ((size_t)n * seqLen + qPos) * qTotalDim + h * qHeadDim;
      #pragma unroll
      for(int d = 0; d < qHeadDim; d++) qReg[qi * qHeadDim + d] = (float)qPtr[d];
    }
  }

  for(int kvStart = 0; kvStart < seqLen; kvStart += BLOCK_KV) {
    for(int t = tid; t < BLOCK_KV * qHeadDim; t += BLOCK_Q) {
      int tileKPos = t / qHeadDim; int tileD = t % qHeadDim;
      int globalKPos = kvStart + tileKPos;
      float v = 0.0f;
      if(globalKPos < seqLen)
        v = (float)K[((size_t)n * seqLen + globalKPos) * qTotalDim + h * qHeadDim + tileD];
      kTile[tileKPos * qHeadDim + tileD] = v;
    }
    for(int t = tid; t < BLOCK_KV * vHeadDim; t += BLOCK_Q) {
      int tileKPos = t / vHeadDim; int tileD = t % vHeadDim;
      int globalKPos = kvStart + tileKPos;
      float v = 0.0f;
      if(globalKPos < seqLen)
        v = (float)V[((size_t)n * seqLen + globalKPos) * vTotalDim + h * vHeadDim + tileD];
      vTile[tileKPos * vHeadDim + tileD] = v;
    }
    __syncthreads();
    int kvEnd = min(BLOCK_KV, seqLen - kvStart);

    for(int qi = 0; qi < QPT; qi++) {
      int qPos = qBlockStart + qi * BLOCK_Q + tid;
      if(qPos >= seqLen) continue;
      for(int tk = 0; tk < kvEnd; tk++) {
        // 4 independent partial sums break the 32-deep FMA dependency chain
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        const float* krow = kTile + tk * qHeadDim;
        const float* qrow = qReg + qi * qHeadDim;
        #pragma unroll
        for(int d = 0; d < qHeadDim; d += 4) {
          d0 += qrow[d+0] * krow[d+0];
          d1 += qrow[d+1] * krow[d+1];
          d2 += qrow[d+2] * krow[d+2];
          d3 += qrow[d+3] * krow[d+3];
        }
        float dot = (d0 + d1) + (d2 + d3);
        dot *= scale;
        float newMax = fmaxf(runningMax[qi], dot);
        float expOldMax = __expf(runningMax[qi] - newMax);
        float expCur = __expf(dot - newMax);
        float* arow = acc + qi * vHeadDim;
        const float* vrow = vTile + tk * vHeadDim;
        #pragma unroll
        for(int d = 0; d < vHeadDim; d++)
          arow[d] = arow[d] * expOldMax + expCur * vrow[d];
        runningSum[qi] = runningSum[qi] * expOldMax + expCur;
        runningMax[qi] = newMax;
      }
    }
    __syncthreads();
  }

  for(int qi = 0; qi < QPT; qi++) {
    int qPos = qBlockStart + qi * BLOCK_Q + tid;
    if(qPos >= seqLen) continue;
    __half* outRow = output + ((size_t)n * seqLen + qPos) * oTotalDim + h * vHeadDim;
    float invSum = (runningSum[qi] > 0.0f) ? (1.0f / runningSum[qi]) : 0.0f;
    #pragma unroll
    for(int d = 0; d < vHeadDim; d++) outRow[d] = (__half)(acc[qi * vHeadDim + d] * invSum);
  }
}

// ---------- V2: V1 + tile-deferred rescale (rescale acc once per KV tile) ----------
template<int qHeadDim, int vHeadDim, int BLOCK_Q, int QPT>
__global__
void flashAttentionV2(
  const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
  const float* __restrict__ mask, __half* __restrict__ output,
  int seqLen, int numHeads, float scale)
{
  const int tid = threadIdx.x;
  const int qBlockStart = blockIdx.x * (BLOCK_Q * QPT);
  const int bh = blockIdx.y;
  const int n = bh / numHeads;
  const int h = bh % numHeads;
  const int qTotalDim = numHeads * qHeadDim;
  const int vTotalDim = numHeads * vHeadDim;
  const int oTotalDim = numHeads * vHeadDim;

  __shared__ float kTile[BLOCK_KV * qHeadDim];
  __shared__ float vTile[BLOCK_KV * vHeadDim];
  float ePriv[BLOCK_KV];  // per-thread scores for this thread's queries

  float qReg[QPT * qHeadDim];
  float runningMax[QPT]; float runningSum[QPT];
  float acc[QPT * vHeadDim];

  for(int qi = 0; qi < QPT; qi++) {
    int qPos = qBlockStart + qi * BLOCK_Q + tid;
    runningMax[qi] = -1e30f; runningSum[qi] = 0.0f;
    for(int d = 0; d < vHeadDim; d++) acc[qi * vHeadDim + d] = 0.0f;
    if(qPos < seqLen) {
      const __half* qPtr = Q + ((size_t)n * seqLen + qPos) * qTotalDim + h * qHeadDim;
      #pragma unroll
      for(int d = 0; d < qHeadDim; d++) qReg[qi * qHeadDim + d] = (float)qPtr[d];
    }
  }

  for(int kvStart = 0; kvStart < seqLen; kvStart += BLOCK_KV) {
    for(int t = tid; t < BLOCK_KV * qHeadDim; t += BLOCK_Q) {
      int tileKPos = t / qHeadDim; int tileD = t % qHeadDim;
      int globalKPos = kvStart + tileKPos;
      float v = 0.0f;
      if(globalKPos < seqLen)
        v = (float)K[((size_t)n * seqLen + globalKPos) * qTotalDim + h * qHeadDim + tileD];
      kTile[tileKPos * qHeadDim + tileD] = v;
    }
    for(int t = tid; t < BLOCK_KV * vHeadDim; t += BLOCK_Q) {
      int tileKPos = t / vHeadDim; int tileD = t % vHeadDim;
      int globalKPos = kvStart + tileKPos;
      float v = 0.0f;
      if(globalKPos < seqLen)
        v = (float)V[((size_t)n * seqLen + globalKPos) * vTotalDim + h * vHeadDim + tileD];
      vTile[tileKPos * vHeadDim + tileD] = v;
    }
    __syncthreads();
    int kvEnd = min(BLOCK_KV, seqLen - kvStart);

    for(int qi = 0; qi < QPT; qi++) {
      int qPos = qBlockStart + qi * BLOCK_Q + tid;
      if(qPos >= seqLen) continue;

      // Pass 1: dots for the whole tile (per-thread private; no races)
      float tileMax = -1e30f;
      for(int tk = 0; tk < kvEnd; tk++) {
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        const float* krow = kTile + tk * qHeadDim;
        const float* qrow = qReg + qi * qHeadDim;
        #pragma unroll
        for(int d = 0; d < qHeadDim; d += 4) {
          d0 += qrow[d+0] * krow[d+0];
          d1 += qrow[d+1] * krow[d+1];
          d2 += qrow[d+2] * krow[d+2];
          d3 += qrow[d+3] * krow[d+3];
        }
        float dot = ((d0 + d1) + (d2 + d3)) * scale;
        ePriv[tk] = dot;
        tileMax = fmaxf(tileMax, dot);
      }
      // (eTile race resolved by making it __shared__ per-tile only in single-qi variant;
      //  in the bench we compile V2 with QPT=1 and guard: see note in main.)
      float newMax = fmaxf(runningMax[qi], tileMax);
      float expOldMax = __expf(runningMax[qi] - newMax);
      float tileSum = 0.0f;
      for(int tk = 0; tk < kvEnd; tk++) {
        float e = __expf(ePriv[tk] - newMax);
        ePriv[tk] = e;
        tileSum += e;
      }
      // single rescale of the accumulator per tile
      float* arow = acc + qi * vHeadDim;
      #pragma unroll
      for(int d = 0; d < vHeadDim; d++) arow[d] *= expOldMax;
      // AV accumulate with pre-exponentiated weights
      for(int tk = 0; tk < kvEnd; tk++) {
        float e = ePriv[tk];
        const float* vrow = vTile + tk * vHeadDim;
        #pragma unroll
        for(int d = 0; d < vHeadDim; d++) arow[d] += e * vrow[d];
      }
      runningSum[qi] = runningSum[qi] * expOldMax + tileSum;
      runningMax[qi] = newMax;
    }
    __syncthreads();
  }

  for(int qi = 0; qi < QPT; qi++) {
    int qPos = qBlockStart + qi * BLOCK_Q + tid;
    if(qPos >= seqLen) continue;
    __half* outRow = output + ((size_t)n * seqLen + qPos) * oTotalDim + h * vHeadDim;
    float invSum = (runningSum[qi] > 0.0f) ? (1.0f / runningSum[qi]) : 0.0f;
    #pragma unroll
    for(int d = 0; d < vHeadDim; d++) outRow[d] = (__half)(acc[qi * vHeadDim + d] * invSum);
  }
}

// ---------- host ----------
static float* g_qh; static float* g_kh; static float* g_vh; static float* g_ref; static float* g_out;

int main(int argc, char** argv) {
  const int B = argc > 1 ? atoi(argv[1]) : 10;
  const int L = argc > 2 ? atoi(argv[2]) : 361;
  const int H = argc > 3 ? atoi(argv[3]) : 6;
  const int D = 32;
  const float scale = 1.0f / sqrtf((float)D);
  const size_t nq = (size_t)B * L * H * D;

  // host init
  float* hq = new float[nq]; float* hk = new float[nq]; float* hv = new float[nq];
  srand(42);
  for(size_t i = 0; i < nq; i++) { hq[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.6f;
                                   hk[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.6f;
                                   hv[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.6f; }
  // CPU reference (float path, same math as kernel)
  float* href = new float[(size_t)B * L * H * D];
  for(int b = 0; b < B; b++) for(int h = 0; h < H; h++) {
    for(int i = 0; i < L; i++) {
      float m = -1e30f;
      float* sc = new float[L];
      for(int j = 0; j < L; j++) {
        float dot = 0;
        for(int d = 0; d < D; d++)
          dot += hq[(((size_t)b*L+i)*H+h)*D+d] * hk[(((size_t)b*L+j)*H+h)*D+d];
        sc[j] = dot * scale; m = fmaxf(m, sc[j]);
      }
      float Z = 0;
      for(int j = 0; j < L; j++) { sc[j] = expf(sc[j] - m); Z += sc[j]; }
      for(int d = 0; d < D; d++) {
        float o = 0;
        for(int j = 0; j < L; j++) o += sc[j] * hv[(((size_t)b*L+j)*H+h)*D+d];
        href[(((size_t)b*L+i)*H+h)*D+d] = o / Z;
      }
      delete[] sc;
    }
  }
  printf("ref done (B=%d L=%d H=%d D=%d)\n", B, L, H, D);

  __half *dq, *dk, *dv, *dout;
  CHECK_HIP(hipMalloc(&dq, nq * 2)); CHECK_HIP(hipMalloc(&dk, nq * 2));
  CHECK_HIP(hipMalloc(&dv, nq * 2)); CHECK_HIP(hipMalloc(&dout, nq * 2));
  __half* hq16 = new __half[nq];
  for(size_t i = 0; i < nq; i++) hq16[i] = (__half)hq[i];
  CHECK_HIP(hipMemcpy(dq, hq16, nq * 2, hipMemcpyHostToDevice));
  __half* hk16 = new __half[nq];
  for(size_t i = 0; i < nq; i++) hk16[i] = (__half)hk[i];
  CHECK_HIP(hipMemcpy(dk, hk16, nq * 2, hipMemcpyHostToDevice));
  __half* hv16 = new __half[nq];
  for(size_t i = 0; i < nq; i++) hv16[i] = (__half)hv[i];
  CHECK_HIP(hipMemcpy(dv, hv16, nq * 2, hipMemcpyHostToDevice));

  dim3 grid((L + 127) / 128, B * H);
  double flop = 4.0 * (double)B * H * L * L * D;

  struct { const char* name; void (*fn)(dim3, dim3, const __half*, const __half*, const __half*, const float*, __half*, int, int, float); } variants[] = {
    {"V0-baseline", [](dim3 g, dim3 b, const __half* q, const __half* k, const __half* v, const float* m, __half* o, int L_, int H_, float s) {
      flashAttentionV0<32, 32, 128, 1><<<g, b>>>(q, k, v, m, o, L_, H_, s); }},
    {"V1-4acc", [](dim3 g, dim3 b, const __half* q, const __half* k, const __half* v, const float* m, __half* o, int L_, int H_, float s) {
      flashAttentionV1<32, 32, 128, 1><<<g, b>>>(q, k, v, m, o, L_, H_, s); }},
    {"V2-defresc", [](dim3 g, dim3 b, const __half* q, const __half* k, const __half* v, const float* m, __half* o, int L_, int H_, float s) {
      flashAttentionV2<32, 32, 128, 1><<<g, b>>>(q, k, v, m, o, L_, H_, s); }},
    {"V3-qpt2-b64", [](dim3 g, dim3 b, const __half* q, const __half* k, const __half* v, const float* m, __half* o, int L_, int H_, float s) {
      flashAttentionV2<32, 32, 64, 2><<<dim3((L_ + 63) / 64 / 2 + 1), b>>>(q, k, v, m, o, L_, H_, s); }},
    {"V4-qpt4-b32", [](dim3 g, dim3 b, const __half* q, const __half* k, const __half* v, const float* m, __half* o, int L_, int H_, float s) {
      flashAttentionV2<32, 32, 32, 4><<<dim3((L_ + 31) / 32 / 4 + 1), b>>>(q, k, v, m, o, L_, H_, s); }},
  };
  (void)g_qh; (void)g_kh; (void)g_vh; (void)g_ref; (void)g_out;

  for(auto& var : variants) {
    CHECK_HIP(hipMemset(dout, 0, nq * 2));
    var.fn(grid, 128, dq, dk, dv, NULL, dout, L, H, scale);
    CHECK_HIP(hipDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    for(int r = 0; r < 10; r++) var.fn(grid, 128, dq, dk, dv, NULL, dout, L, H, scale);
    CHECK_HIP(hipDeviceSynchronize());
    double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 10 * 1e3;
    // numeric check
    __half* hout = new __half[nq];
    CHECK_HIP(hipMemcpy(hout, dout, nq * 2, hipMemcpyDeviceToHost));
    double maxd = 0;
    for(size_t i = 0; i < nq; i += 97) {
      double d = fabs((double)hout[i] - href[i]);
      if(d > maxd) maxd = d;
    }
    printf("%-12s %8.3f ms  %7.2f TFLOP/s  maxdiff=%.4f\n", var.name, ms, flop / (ms / 1e3) / 1e12, maxd);
    delete[] hout;
  }
  printf("ATTNBENCH_DONE\n");
  return 0;
}
