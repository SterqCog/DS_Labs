#include "matmul.h"
#include <immintrin.h>
#include <algorithm>

void matmul_avx2(int N,
    const std::vector<std::complex<float>>& A,
    const std::vector<std::complex<float>>& B,
    std::vector<std::complex<float>>& C)
{
    const int BLOCK_SIZE = 64;

    // Извлекаем сырые указатели на float из векторов
    const float* A_f = reinterpret_cast<const float*>(A.data());
    const float* B_f = reinterpret_cast<const float*>(B.data());
    float* C_f = reinterpret_cast<float*>(C.data());

    // Обнуляем матрицу C локально перед расчетами (опционально, но безопасно)
    std::fill(C_f, C_f + 2 * N * N, 0.0f);

    __m256 sign_mask = _mm256_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);

#pragma omp parallel for schedule(guided)
    for (int si = 0; si < N; si += BLOCK_SIZE) {
        for (int sk = 0; sk < N; sk += BLOCK_SIZE) {
            for (int sj = 0; sj < N; sj += BLOCK_SIZE) {

                for (int i = si; i < si + BLOCK_SIZE; i += 4) {
                    for (int k = sk; k < std::min(sk + BLOCK_SIZE, N); ++k) {

                        float ar0 = A_f[2 * (i * N + k)];
                        float ai0 = A_f[2 * (i * N + k) + 1];
                        __m256 vec_ar0 = _mm256_set1_ps(ar0);
                        __m256 vec_ai0 = _mm256_set1_ps(ai0);

                        float ar1 = A_f[2 * ((i + 1) * N + k)];
                        float ai1 = A_f[2 * ((i + 1) * N + k) + 1];
                        __m256 vec_ar1 = _mm256_set1_ps(ar1);
                        __m256 vec_ai1 = _mm256_set1_ps(ai1);

                        float ar2 = A_f[2 * ((i + 2) * N + k)];
                        float ai2 = A_f[2 * ((i + 2) * N + k) + 1];
                        __m256 vec_ar2 = _mm256_set1_ps(ar2);
                        __m256 vec_ai2 = _mm256_set1_ps(ai2);

                        float ar3 = A_f[2 * ((i + 3) * N + k)];
                        float ai3 = A_f[2 * ((i + 3) * N + k) + 1];
                        __m256 vec_ar3 = _mm256_set1_ps(ar3);
                        __m256 vec_ai3 = _mm256_set1_ps(ai3);

                        int j_limit = std::min(sj + BLOCK_SIZE, N);

                        for (int j = sj; j < j_limit; j += 4) {
                            int b_idx = 2 * (k * N + j);

                            __m256 vec_b = _mm256_loadu_ps(&B_f[b_idx]);
                            __m256 vec_b_shuf = _mm256_shuffle_ps(vec_b, vec_b, _MM_SHUFFLE(2, 3, 0, 1));

                            // --- СТРОКА 0 ---
                            int c_idx0 = 2 * (i * N + j);
                            __m256 vec_c0 = _mm256_loadu_ps(&C_f[c_idx0]);
                            vec_c0 = _mm256_fmadd_ps(vec_ar0, vec_b, vec_c0);
                            __m256 i_term0 = _mm256_mul_ps(vec_ai0, vec_b_shuf);
                            i_term0 = _mm256_xor_ps(i_term0, sign_mask);
                            vec_c0 = _mm256_add_ps(vec_c0, i_term0);
                            _mm256_storeu_ps(&C_f[c_idx0], vec_c0);

                            // --- СТРОКА 1 ---
                            int c_idx1 = 2 * ((i + 1) * N + j);
                            __m256 vec_c1 = _mm256_loadu_ps(&C_f[c_idx1]);
                            vec_c1 = _mm256_fmadd_ps(vec_ar1, vec_b, vec_c1);
                            __m256 i_term1 = _mm256_mul_ps(vec_ai1, vec_b_shuf);
                            i_term1 = _mm256_xor_ps(i_term1, sign_mask);
                            vec_c1 = _mm256_add_ps(vec_c1, i_term1);
                            _mm256_storeu_ps(&C_f[c_idx1], vec_c1);

                            // --- СТРОКА 2 ---
                            int c_idx2 = 2 * ((i + 2) * N + j);
                            __m256 vec_c2 = _mm256_loadu_ps(&C_f[c_idx2]);
                            vec_c2 = _mm256_fmadd_ps(vec_ar2, vec_b, vec_c2);
                            __m256 i_term2 = _mm256_mul_ps(vec_ai2, vec_b_shuf);
                            i_term2 = _mm256_xor_ps(i_term2, sign_mask);
                            vec_c2 = _mm256_add_ps(vec_c2, i_term2);
                            _mm256_storeu_ps(&C_f[c_idx2], vec_c2);

                            // --- СТРОКА 3 ---
                            int c_idx3 = 2 * ((i + 3) * N + j);
                            __m256 vec_c3 = _mm256_loadu_ps(&C_f[c_idx3]);
                            vec_c3 = _mm256_fmadd_ps(vec_ar3, vec_b, vec_c3);
                            __m256 i_term3 = _mm256_mul_ps(vec_ai3, vec_b_shuf);
                            i_term3 = _mm256_xor_ps(i_term3, sign_mask);
                            vec_c3 = _mm256_add_ps(vec_c3, i_term3);
                            _mm256_storeu_ps(&C_f[c_idx3], vec_c3);
                        }
                    }
                }
            }
        }
    }
}