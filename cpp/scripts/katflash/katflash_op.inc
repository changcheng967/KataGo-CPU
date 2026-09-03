// KatFlashAttention op - shared by katflash.cpp (full-model harness) and
// test_layer.cpp (single-layer A/B).
//
// Fused KataGo transformer attention as one OV custom op: head split, RoPE on q
// and k, scores GEMM, scale, board mask, softmax, probs@V, head merge.
//
// v1 kernel (AVX-512, dh == 32): K is transposed to [dh][Lp] so one 32-FMA
// chain yields 16 scores per key-tile (no per-pair horizontal reduce, which a
// naive per-(q,j) dot would need 5:1 extra instructions for at this depth);
// V stays row-major [Lp][dh] so probs@V accumulates with 2 FMAs per key.
// Softmax runs fully in registers (L padded to a multiple of 16) with a
// polynomial 16-wide exp. v0 (portable scalar) is kept as the generic fallback.
#pragma once
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "openvino/openvino.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/op/op.hpp"

namespace {
// KATAGO_KATFLASH_PROF=1: per-phase cycle counters, dumped at process exit.
inline bool katFlashProfOn() {
  static const bool on = std::getenv("KATAGO_KATFLASH_PROF") != NULL;
  return on;
}
inline uint64_t katFlashClk() { return __rdtsc(); }
inline void katFlashAdd(std::atomic<uint64_t>& c, uint64_t d) {
  c.fetch_add(d, std::memory_order_relaxed);
}
std::atomic<uint64_t> g_kfBuild{0}, g_kfRopeQ{0}, g_kfScores{0}, g_kfSoftmax{0}, g_kfAV{0}, g_kfCalls{0};
struct KatFlashProfReport {
  ~KatFlashProfReport() {
    if(g_kfCalls.load() == 0) return;
    fprintf(stderr, "[katflash] calls=%llu build=%llu ropeQ=%llu scores=%llu softmax=%llu av=%llu (cycles)\n",
            (unsigned long long)g_kfCalls.load(), (unsigned long long)g_kfBuild.load(),
            (unsigned long long)g_kfRopeQ.load(), (unsigned long long)g_kfScores.load(),
            (unsigned long long)g_kfSoftmax.load(), (unsigned long long)g_kfAV.load());
  }
};
KatFlashProfReport g_katFlashProfReportInstance;


inline bool katFlashAvx512Supported() {
  static const bool ok = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma");
  return ok;
}

// 16-wide exp: 2^(x*log2e) via degree-5 poly on the fraction + exponent bits.
// Inputs are <= 0 here (post max-subtraction); clamp to -87 to flush to ~0.
inline __m512 katExpPs(__m512 x) {
  const __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
  const __m512 lo = _mm512_set1_ps(-87.0f);
  x = _mm512_max_ps(x, lo);
  __m512 t = _mm512_mul_ps(x, log2e);
  // n = round(t), f = t - n
  __m512 n = _mm512_cvtepi32_ps(_mm512_cvtps_epi32(t));  // round-to-nearest
  __m512 f = _mm512_sub_ps(t, n);
  // 2^f poly (Cephes deg-5, ~1e-7 rel)
  __m512 p = _mm512_set1_ps(0.00133335581f);
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.00961812911f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.0555041102f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.240226509f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.693147078f));
  p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.000000018f));
  // 2^n by exponent addition: n + 127 into the exponent field
  __m512i ni = _mm512_cvtps_epi32(n);
  __m512 two_n = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(ni, _mm512_set1_epi32(127)), 23));
  return _mm512_mul_ps(p, two_n);
}

struct KatFlashAttention : public ov::op::Op {
  OPENVINO_OP("KatFlashAttention", "org.katago");

  KatFlashAttention() = default;
  KatFlashAttention(const ov::OutputVector& args) : ov::op::Op(args) {
    constructor_validate_and_infer_types();
  }

  std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_inputs) const override {
    return std::make_shared<KatFlashAttention>(new_inputs);
  }
  bool has_evaluate() const override { return true; }

  void validate_and_infer_types() override {
    auto rc = get_input_partial_shape(4);
    int64_t H = rc[1].get_length(), dh = rc[3].get_length();
    auto qs = get_input_partial_shape(0);  // [N,19,19,C] or [N,L,C]
    OPENVINO_ASSERT(qs[qs.size() - 1].is_dynamic() || qs[qs.size() - 1].get_length() == H * dh,
                    "KatFlashAttention: channel mismatch");
    set_output_type(0, ov::element::f32, qs);
  }

  bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
    if(inputs.size() != 8)
      return false;
    const ov::Tensor& tq = inputs[0];
    const ov::Tensor& tk = inputs[1];
    const ov::Tensor& tv = inputs[2];
    const ov::Tensor& tmask = inputs[3];
    const ov::Tensor& tcos = inputs[4];
    const ov::Tensor& tsin = inputs[5];
    const ov::Tensor& tswap = inputs[6];
    const ov::Tensor& tscale = inputs[7];

    const ov::Shape& sq = tq.get_shape();
    int64_t N = sq[0], C = sq[sq.size() - 1];
    int64_t L = 1;
    for(size_t i = 1; i + 1 < sq.size(); i++) L *= (int64_t)sq[i];
    const ov::Shape& sc = tcos.get_shape();
    int64_t H = sc[1], dh = sc[3], Lc = sc[2];

    if(outputs[0].get_element_type() != ov::element::f32 || outputs[0].get_shape() != sq)
      outputs[0] = ov::Tensor(ov::element::f32, sq);
    float* pout = outputs[0].data<float>();

    int swapI[64];
    {
      const int64_t* p64 = nullptr;
      const int32_t* p32 = nullptr;
      if(tswap.get_element_type() == ov::element::i64) p64 = tswap.data<const int64_t>();
      else p32 = tswap.data<const int32_t>();
      for(int64_t d = 0; d < dh && d < 64; d++)
        swapI[d] = (int)(p64 ? p64[d] : p32[d]);
    }

    const float* pq = tq.data<const float>();
    const float* pk = tk.data<const float>();
    const float* pv = tv.data<const float>();
    const float* pmask = tmask.data<const float>();
    const float* pcos = tcos.data<const float>();
    const float* psin = tsin.data<const float>();
    const float scale = tscale.data<const float>()[0];

#if defined(__x86_64__) || defined(_M_X64)
    if(katFlashAvx512Supported() && dh == 32 && Lc == L)
      evalV1(pq, pk, pv, pmask, pcos, psin, swapI, scale, N, H, dh, L, C, pout);
    else
#endif
      evalV0(pq, pk, pv, pmask, pcos, psin, swapI, scale, N, H, dh, L, C, pout);
    return true;
  }

  // v1: AVX-512, dh == 32. K transposed to [32][Lp] (one 32-FMA chain per key
  // tile yields 16 scores), V row-major [Lp][32] (2 FMAs per key for probs@V).
  void evalV1(const float* pq, const float* pk, const float* pv, const float* pmask,
              const float* pcos, const float* psin, const int* swapI, float scale,
              int64_t N, int64_t H, int64_t dh, int64_t L, int64_t C, float* pout) const {
    const int Lp = (int)((L + 15) & ~15);  // 361 -> 368
    const int nTiles = Lp / 16;            // 23
    ov::parallel_for(N * H, [&](int64_t task) {
      const int64_t n = task / H, h = task % H;
      const float* q = pq + n * L * C + h * dh;
      const float* k = pk + n * L * C + h * dh;
      const float* v = pv + n * L * C + h * dh;
      const float* cosT = pcos + h * L * dh;
      const float* sinT = psin + h * L * dh;
      const float* mask = pmask + n * L;
      float* out = pout + n * L * C + h * dh;
      const int64_t hStep = C;

      // Thread-local persistent scratch: kt/vtT [32][Lp] dim-major (rope'd keys,
      // values), krow/vrow [L][32] contiguous staging, maskP, probs.
      thread_local std::vector<float> kt_v, vtT_v, krow_v, vrow_v, maskP_v, probs_v;
      kt_v.assign((size_t)32 * Lp, 0.0f);
      vtT_v.assign((size_t)32 * Lp, 0.0f);
      krow_v.assign((size_t)L * 32, 0.0f);
      vrow_v.assign((size_t)L * 32, 0.0f);
      maskP_v.assign(Lp, -1e30f);
      probs_v.assign(Lp, 0.0f);
      float* kt = kt_v.data();
      float* vtT = vtT_v.data();
      float* krow = krow_v.data();
      float* vrow = vrow_v.data();
      float* maskP = maskP_v.data();
      float* probs = probs_v.data();

      const bool kfProf = katFlashProfOn();
      uint64_t kfT0 = kfProf ? katFlashClk() : 0;
      // stage rope'd K and raw V contiguously (row-major reads and writes)
      for(int j = 0; j < L; j++) {
        const float* kj = k + j * hStep;
        const float* cj = cosT + j * dh;
        const float* sj = sinT + j * dh;
        float* kr = krow + (size_t)j * 32;
        for(int d = 0; d < 32; d++)
          kr[d] = kj[d] * cj[d] + kj[swapI[d]] * sj[d];
        const float* vj = v + j * hStep;
        float* vr = vrow + (size_t)j * 32;
        for(int d = 0; d < 32; d++)
          vr[d] = vj[d];
        maskP[j] = mask[j];
      }
      // blocked transposes krow/vrow [L][32] -> kt/vtT [32][Lp]
      for(int jj = 0; jj < L; jj += 16) {
        const int jmax = (jj + 16 < L) ? jj + 16 : L;
        for(int dd = 0; dd < 32; dd += 16)
          for(int j = jj; j < jmax; j++) {
            for(int d = dd; d < dd + 16; d++) {
              kt[(size_t)d * Lp + j] = krow[(size_t)j * 32 + d];
              vtT[(size_t)d * Lp + j] = vrow[(size_t)j * 32 + d];
            }
          }
      }

      if(kfProf) { katFlashAdd(g_kfBuild, katFlashClk() - kfT0); katFlashAdd(g_kfCalls, 1); }
      uint64_t kfSumRope = 0, kfSumScores = 0, kfSumSoft = 0, kfSumAV = 0;

      // Query-tiled main loop: process 16 queries per pass so kt/vtT stream from
      // L2 once per tile-of-16 instead of once per query (the per-query version
      // is L2-bandwidth-bound at ~47KB of key/value traffic per query).
      thread_local std::vector<float> qr16_v, sbuf_v;
      qr16_v.assign((size_t)16 * 32, 0.0f);   // rope'd queries [16][32]
      sbuf_v.assign((size_t)16 * Lp, 0.0f);   // scores, then probs [16][Lp]
      float* qr16 = qr16_v.data();
      float* sbuf = sbuf_v.data();

      const __m512 vzero = _mm512_setzero_ps();
      const __m512 vscale = _mm512_set1_ps(scale);
      for(int qi = 0; qi < L; qi += 16) {
        const int Q = (qi + 16 <= L) ? 16 : (L - qi);

        uint64_t kfT;
        if(kfProf) kfT = katFlashClk();
        // rope Q queries into qr16 (contiguous staging)
        for(int qq = 0; qq < Q; qq++) {
          const float* qsrc = q + (size_t)(qi + qq) * hStep;
          const float* ci = cosT + (qi + qq) * dh;
          const float* si = sinT + (qi + qq) * dh;
          float* dst = qr16 + (size_t)qq * 32;
          for(int d = 0; d < 32; d++)
            dst[d] = qsrc[d] * ci[d] + qsrc[swapI[d]] * si[d];
        }

        if(kfProf) { kfSumRope += katFlashClk() - kfT; kfT = katFlashClk(); }
        // scores: kt row loaded once per (tile,d) and FMAd against all Q queries
        for(int t = 0; t < nTiles; t++) {
          __m512 Srow[16];
          for(int q = 0; q < Q; q++) Srow[q] = vzero;
          const __m512 mt = _mm512_loadu_ps(&maskP[t * 16]);
          for(int d = 0; d < 32; d++) {
            const __m512 ktr = _mm512_loadu_ps(&kt[(size_t)d * Lp + t * 16]);
            for(int q = 0; q < Q; q++)
              Srow[q] = _mm512_fmadd_ps(_mm512_set1_ps(qr16[(size_t)q * 32 + d]), ktr, Srow[q]);
          }
          for(int q = 0; q < Q; q++)
            _mm512_storeu_ps(&sbuf[(size_t)q * Lp + t * 16], _mm512_add_ps(_mm512_mul_ps(Srow[q], vscale), mt));
        }

        if(kfProf) { kfSumScores += katFlashClk() - kfT; kfT = katFlashClk(); }
        // softmax per query, in place
        for(int q = 0; q < Q; q++) {
          float* sq = sbuf + (size_t)q * Lp;
          __m512 mvec = _mm512_loadu_ps(sq);
          for(int t = 1; t < nTiles; t++) mvec = _mm512_max_ps(mvec, _mm512_loadu_ps(sq + t * 16));
          const float m = _mm512_reduce_max_ps(mvec);
          const __m512 mb = _mm512_set1_ps(m);
          __m512 zvec = vzero;
          for(int t = 0; t < nTiles; t++) {
            __m512 e = katExpPs(_mm512_sub_ps(_mm512_loadu_ps(sq + t * 16), mb));
            _mm512_storeu_ps(sq + t * 16, e);
            zvec = _mm512_add_ps(zvec, e);
          }
          const __m512 iz = _mm512_set1_ps(1.0f / (_mm512_reduce_add_ps(zvec) + 1e-30f));
          for(int t = 0; t < nTiles; t++)
            _mm512_storeu_ps(sq + t * 16, _mm512_mul_ps(_mm512_loadu_ps(sq + t * 16), iz));
        }

        if(kfProf) { kfSumSoft += katFlashClk() - kfT; kfT = katFlashClk(); }
        // probs @ V: vtT row loaded once per (dim,tile), FMAd against the
        // queries' probs; one horizontal reduce per (q,dim) at the end.
        // Queries are processed in chunks of 8 so the accumulator array fits
        // in registers (a live acc[16] plus row vectors spills to stack).
        for(int d = 0; d < 32; d++) {
          for(int q0 = 0; q0 < Q; q0 += 8) {
            const int qc = (q0 + 8 <= Q) ? 8 : (Q - q0);
            __m512 acc[8];
            for(int q = 0; q < qc; q++) acc[q] = vzero;
            for(int t = 0; t < nTiles; t++) {
              const __m512 vrow = _mm512_loadu_ps(&vtT[(size_t)d * Lp + t * 16]);
              for(int q = 0; q < qc; q++)
                acc[q] = _mm512_fmadd_ps(_mm512_loadu_ps(&sbuf[(size_t)(q0 + q) * Lp + t * 16]), vrow, acc[q]);
            }
            for(int q = 0; q < qc; q++)
              out[(size_t)(qi + q0 + q) * hStep + d] = _mm512_reduce_add_ps(acc[q]);
          }
        }
        if(kfProf) kfSumAV += katFlashClk() - kfT;
      }
      if(kfProf) {
        katFlashAdd(g_kfRopeQ, kfSumRope);
        katFlashAdd(g_kfScores, kfSumScores);
        katFlashAdd(g_kfSoftmax, kfSumSoft);
        katFlashAdd(g_kfAV, kfSumAV);
      }
    });
  }

  // v0: portable scalar fallback (also the semantic reference)
  void evalV0(const float* pq, const float* pk, const float* pv, const float* pmask,
              const float* pcos, const float* psin, const int* swapI, float scale,
              int64_t N, int64_t H, int64_t dh, int64_t L, int64_t C, float* pout) const {
    ov::parallel_for(N * H, [&](int64_t task) {
      const int64_t n = task / H, h = task % H;
      const float* q = pq + n * L * C + h * dh;
      const float* k = pk + n * L * C + h * dh;
      const float* v = pv + n * L * C + h * dh;
      const float* cosT = pcos + h * L * dh;
      const float* sinT = psin + h * L * dh;
      const float* mask = pmask + n * L;
      float* out = pout + n * L * C + h * dh;
      const int64_t hStep = C;

      std::vector<float> krot(L * dh);
      for(int64_t j = 0; j < L; j++) {
        const float* kj = k + j * hStep;
        const float* cj = cosT + j * dh;
        const float* sj = sinT + j * dh;
        float* r = krot.data() + j * dh;
        for(int64_t d = 0; d < dh; d++)
          r[d] = kj[d] * cj[d] + kj[swapI[d]] * sj[d];
      }

      std::vector<float> qrot(dh), s(L), o(dh);
      for(int64_t i = 0; i < L; i++) {
        const float* qi = q + i * hStep;
        const float* ci = cosT + i * dh;
        const float* si = sinT + i * dh;
        for(int64_t d = 0; d < dh; d++)
          qrot[d] = qi[d] * ci[d] + qi[swapI[d]] * si[d];
        float m = -1e30f;
        for(int64_t j = 0; j < L; j++) {
          const float* r = krot.data() + j * dh;
          float acc = 0.f;
          for(int64_t d = 0; d < dh; d++)
            acc += qrot[d] * r[d];
          float sv = acc * scale + mask[j];
          s[j] = sv;
          if(sv > m) m = sv;
        }
        float Z = 0.f;
        for(int64_t j = 0; j < L; j++) {
          float e = std::exp(s[j] - m);
          s[j] = e;
          Z += e;
        }
        float invZ = 1.0f / Z;
        for(int64_t d = 0; d < dh; d++) o[d] = 0.f;
        for(int64_t j = 0; j < L; j++) {
          const float p = s[j] * invZ;
          const float* vj = v + j * hStep;
          for(int64_t d = 0; d < dh; d++)
            o[d] += p * vj[d];
        }
        float* oi = out + i * hStep;
        for(int64_t d = 0; d < dh; d++)
          oi[d] = o[d];
      }
    });
  }
};

}  // namespace
