#include "matmul.h"
#include <immintrin.h>
#include <algorithm>

void matmul_avx512(int N,
    const std::vector<std::complex<float>>& A,
    const std::vector<std::complex<float>>& B,
    std::vector<std::complex<float>>& C)
{
    const int n = N;
    const int BLOCK_SIZE = 64;

    const std::complex<float>* A_flat = A.data();
    const std::complex<float>* B_flat = B.data();
    std::complex<float>* C_flat = C.data();

    std::fill(C_flat, C_flat + n * n, std::complex<float>(0.0f, 0.0f));

    __m512 sign_mask = _mm512_set_ps(
        0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f,
        0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f
    );

#pragma omp parallel for schedule(static)
    for (int si = 0; si < n; si += BLOCK_SIZE) {
        for (int sk = 0; sk < n; sk += BLOCK_SIZE) {
            for (int sj = 0; sj < n; sj += BLOCK_SIZE) {

                for (int i = si; i < si + BLOCK_SIZE; i += 4) {
                    for (int k = sk; k < sk + BLOCK_SIZE; ++k) {

                        __m512 vec_ar0 = _mm512_set1_ps(A_flat[i * n + k].real());
                        __m512 vec_ai0 = _mm512_set1_ps(A_flat[i * n + k].imag());

                        __m512 vec_ar1 = _mm512_set1_ps(A_flat[(i + 1) * n + k].real());
                        __m512 vec_ai1 = _mm512_set1_ps(A_flat[(i + 1) * n + k].imag());

                        __m512 vec_ar2 = _mm512_set1_ps(A_flat[(i + 2) * n + k].real());
                        __m512 vec_ai2 = _mm512_set1_ps(A_flat[(i + 2) * n + k].imag());

                        __m512 vec_ar3 = _mm512_set1_ps(A_flat[(i + 3) * n + k].real());
                        __m512 vec_ai3 = _mm512_set1_ps(A_flat[(i + 3) * n + k].imag());

                        for (int j = sj; j < sj + BLOCK_SIZE; j += 8) {
                            int idx_b = k * n + j;

                            __m512 vec_b = _mm512_loadu_ps(reinterpret_cast<const float*>(&B_flat[idx_b]));
                            __m512 vec_b_shuf = _mm512_shuffle_ps(vec_b, vec_b, _MM_SHUFFLE(2, 3, 0, 1));

                            int idx_c0 = i * n + j;
                            __m512 vec_c0 = _mm512_loadu_ps(reinterpret_cast<float*>(&C_flat[idx_c0]));
                            vec_c0 = _mm512_fmadd_ps(vec_ar0, vec_b, vec_c0);
                            __m512 i_term0 = _mm512_mul_ps(vec_ai0, vec_b_shuf);
                            i_term0 = _mm512_xor_ps(i_term0, sign_mask);
                            vec_c0 = _mm512_add_ps(vec_c0, i_term0);
                            _mm512_storeu_ps(reinterpret_cast<float*>(&C_flat[idx_c0]), vec_c0);

                            int idx_c1 = (i + 1) * n + j;
                            __m512 vec_c1 = _mm512_loadu_ps(reinterpret_cast<float*>(&C_flat[idx_c1]));
                            vec_c1 = _mm512_fmadd_ps(vec_ar1, vec_b, vec_c1);
                            __m512 i_term1 = _mm512_mul_ps(vec_ai1, vec_b_shuf);
                            i_term1 = _mm512_xor_ps(i_term1, sign_mask);
                            vec_c1 = _mm512_add_ps(vec_c1, i_term1);
                            _mm512_storeu_ps(reinterpret_cast<float*>(&C_flat[idx_c1]), vec_c1);

                            int idx_c2 = (i + 2) * n + j;
                            __m512 vec_c2 = _mm512_loadu_ps(reinterpret_cast<float*>(&C_flat[idx_c2]));
                            vec_c2 = _mm512_fmadd_ps(vec_ar2, vec_b, vec_c2);
                            __m512 i_term2 = _mm512_mul_ps(vec_ai2, vec_b_shuf);
                            i_term2 = _mm512_xor_ps(i_term2, sign_mask);
                            vec_c2 = _mm512_add_ps(vec_c2, i_term2);
                            _mm512_storeu_ps(reinterpret_cast<float*>(&C_flat[idx_c2]), vec_c2);

                            int idx_c3 = (i + 3) * n + j;
                            __m512 vec_c3 = _mm512_loadu_ps(reinterpret_cast<float*>(&C_flat[idx_c3]));
                            vec_c3 = _mm512_fmadd_ps(vec_ar3, vec_b, vec_c3);
                            __m512 i_term3 = _mm512_mul_ps(vec_ai3, vec_b_shuf);
                            i_term3 = _mm512_xor_ps(i_term3, sign_mask);
                            vec_c3 = _mm512_add_ps(vec_c3, i_term3);
                            _mm512_storeu_ps(reinterpret_cast<float*>(&C_flat[idx_c3]), vec_c3);
                        }
                    }
                }
            }
        }
    }
}
