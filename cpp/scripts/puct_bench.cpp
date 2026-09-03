// AVX-512 PUCT selection benchmark: scalar vs vectorized
#include <cstdio>
#include <cstdlib>
#include <chrono>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

#define N 362
static float policy[N], utility[N], weightF[N];
static double weightD[N];

int scalarFindBest(double es, int pla) {
    double best = -1e30; int bi = -1;
    double flip = pla == 0 ? 1.0 : -1.0;
    for (int i = 0; i < N; i++) {
        double p = policy[i];
        if (p < 0) continue;
        double sv = es * p / (1.0 + weightD[i]) + flip * utility[i];
        if (sv > best) { best = sv; bi = i; }
    }
    return bi;
}

#ifdef __AVX512F__
int vectorFindBest(float esF, int pla) {
    float flip = pla == 0 ? 1.0f : -1.0f;
    __m512 v_es = _mm512_set1_ps(esF);
    __m512 v_flip = _mm512_set1_ps(flip);
    __m512 v_one = _mm512_set1_ps(1.0f);
    __m512 v_ninf = _mm512_set1_ps(-1e30f);
    __m512 v_zero = _mm512_setzero_ps();
    float bestScore = -1e30f;
    int bestIdx = -1;
    for (int i = 0; i + 16 <= N; i += 16) {
        __m512 p = _mm512_loadu_ps(&policy[i]);
        __m512 u = _mm512_loadu_ps(&utility[i]);
        __m512 w = _mm512_loadu_ps(&weightF[i]);
        __mmask16 valid = _mm512_cmp_ps_mask(p, v_zero, _CMP_GE_OQ);
        __m512 explore = _mm512_mask_div_ps(v_ninf, valid,
            _mm512_mul_ps(v_es, p), _mm512_add_ps(v_one, w));
        __m512 score = _mm512_add_ps(explore, _mm512_mul_ps(v_flip, u));
        float bmax = _mm512_reduce_max_ps(score);
        if (bmax > bestScore) {
            __mmask16 eq = _mm512_cmp_ps_mask(score, _mm512_set1_ps(bmax), _CMP_EQ_OQ);
            bestIdx = i + __builtin_ctz(eq);
            bestScore = bmax;
        }
    }
    for (int i = 352; i < N; i++) {
        float p = policy[i];
        if (p < 0) continue;
        float sv = esF * p / (1.0f + weightF[i]) + flip * utility[i];
        if (sv > bestScore) { bestScore = sv; bestIdx = i; }
    }
    return bestIdx;
}
#endif

int main() {
    srand(42);
    for (int i = 0; i < N; i++) {
        policy[i] = (rand() % 100) < 70 ? (float)rand() / RAND_MAX * 0.05f : -1.0f;
        utility[i] = (float)(rand() - RAND_MAX/2) / (RAND_MAX/2) * 0.3f;
        weightD[i] = (double)(rand() % 200) / 10.0;
        weightF[i] = (float)weightD[i];
    }
    policy[250] = 0.4f; utility[250] = 0.7f; weightD[250] = 3.0; weightF[250] = 3.0f;
    double es = 1.2; int pla = 0;
    int s = scalarFindBest(es, pla);
#ifdef __AVX512F__
    int v = vectorFindBest((float)es, pla);
    printf("scalar=%d vector=%d %s\n", s, v, s == v ? "MATCH" : "MISMATCH");
#endif
    volatile int sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < 2000000; it++) sink += scalarFindBest(es, pla);
    auto t1 = std::chrono::steady_clock::now();
    double ns_s = std::chrono::duration<double, std::nano>(t1 - t0).count() / 2000000;
#ifdef __AVX512F__
    t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < 2000000; it++) sink += vectorFindBest((float)es, pla);
    t1 = std::chrono::steady_clock::now();
    double ns_v = std::chrono::duration<double, std::nano>(t1 - t0).count() / 2000000;
    printf("scalar: %.1f ns  vector: %.1f ns  speedup: %.2fx\n", ns_s, ns_v, ns_s / ns_v);
#endif
    printf("PUCT_BENCH_DONE\n");
    return 0;
}
