#include <complex>
#include <Windows.h>
#include <cblas.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <algorithm>
 
using namespace std;
using namespace std::chrono;
 
double calculateComplexity(int n) {
    return 2 * pow(n, 3);
}
 
double calculatePerformance(double complexity, double timeInSeconds) {
    return complexity / (timeInSeconds * 1e6);
}
 
void multiplyMatrices(const vector<vector<complex<float>>>& A,
    const vector<vector<complex<float>>>& B,
    vector<vector<complex<float>>>& C) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
 
void multiplyMatricesBLAS(const vector<complex<float>>& A,
    const vector<complex<float>>& B,
    vector<complex<float>>& C,
    const int N) {
    const auto* A_data = reinterpret_cast<const float*>(A.data());
    const auto* B_data = reinterpret_cast<const float*>(B.data());
    auto* C_data = reinterpret_cast<float*>(C.data());
 
    const float alpha[2] = { 1.0, 0.0 };
    const float beta[2] = { 0.0, 0.0 };
 
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A_data, N, B_data, N, beta, C_data, N);
}
 


struct Complex {
    float re, im;
};

void multiplyMatricesOptimized(int n, float* A, float* B, float* C) {
    const int BS = 32;

    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < n; ii += BS)
    for (int jj = 0; jj < n; jj += BS)
    for (int kk = 0; kk < n; kk += BS)

    for (int i = ii; i < std::min(ii + BS, n); i++) {
        for (int k = kk; k < std::min(kk + BS, n); k++) {

            float ar = A[2*(i*n + k)];
            float ai = A[2*(i*n + k) + 1];

            __m256 a = _mm256_set_ps(ai, ar, ai, ar, ai, ar, ai, ar);

            int j = jj;

            for (; j + 3 < std::min(jj + BS, n); j += 4) {

                __m256 b = _mm256_loadu_ps(&B[2*(k*n + j)]);
                __m256 c = _mm256_loadu_ps(&C[2*(i*n + j)]);

                __m256 b_perm = _mm256_permute_ps(b, 0xB1);

                __m256 mul1 = _mm256_mul_ps(a, b);
                __m256 mul2 = _mm256_mul_ps(a, b_perm);

                __m256 res = _mm256_addsub_ps(mul1, mul2);

                c = _mm256_add_ps(c, res);

                _mm256_storeu_ps(&C[2*(i*n + j)], c);
            }

            // ?? îñòàòîê
            for (; j < std::min(jj + BS, n); j++) {
                float br = B[2*(k*n + j)];
                float bi = B[2*(k*n + j) + 1];

                C[2*(i*n + j)]     += ar * br - ai * bi;
                C[2*(i*n + j) + 1] += ar * bi + ai * br;
            }
        }
    }
}
 
int main() {
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);
 
    srand(static_cast<unsigned>(time(0)));
 
    int n = 4096;
    vector<vector<complex<float>>> A(n, vector<complex<float>>(n));
    vector<vector<complex<float>>> B(n, vector<complex<float>>(n));
    vector<vector<complex<float>>> C(n, vector<complex<float>>(n));
 
    vector<complex<float>> A_flat(n * n), B_flat(n * n), C_flat(n * n);
 
    for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
        A[i][j] = complex<float>(rand() / float(RAND_MAX), rand() / float(RAND_MAX));
        B[i][j] = complex<float>(rand() / float(RAND_MAX), rand() / float(RAND_MAX));
        A_flat[i*n + j] = A[i][j];
        B_flat[i*n + j] = B[i][j];
    }
 
    multiplyMatrices(A, B, C);
 
    auto start = high_resolution_clock::now();
    multiplyMatrices(A, B, C);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    double timeFirstVariant = duration.count() / 1e6;
 
    start = high_resolution_clock::now();
    multiplyMatricesBLAS(A_flat, B_flat, C_flat, n);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    double timeSecondVariant = duration.count() / 1e6;
 
    start = high_resolution_clock::now();
    vector<float> A_arr(2*n*n), B_arr(2*n*n), C_arr(2*n*n, 0.0f);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            A_arr[2*(i*n+j)] = A[i][j].real();
            A_arr[2*(i*n+j)+1] = A[i][j].imag();
            B_arr[2*(i*n+j)] = B[i][j].real();
            B_arr[2*(i*n+j)+1] = B[i][j].imag();
        }

    multiplyMatricesOptimized(n, A_arr.data(), B_arr.data(), C_arr.data());
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    double timeThirdVariant = duration.count() / 1e6;
 
    double complexity = calculateComplexity(n);
    double performanceFirstVariant = calculatePerformance(complexity, timeFirstVariant);
    double performanceSecondVariant = calculatePerformance(complexity, timeSecondVariant);
    double performanceThirdVariant = calculatePerformance(complexity, timeThirdVariant);
 
    cout << "Ñëîæíîñòü àëãîðèòìà: " << complexity << endl;
    cout << "Ïðîèçâîäèòåëüíîñòü ïåðâîãî âàðèàíòà: " << performanceFirstVariant << " MFlops" << endl;
    cout << "Ïðîèçâîäèòåëüíîñòü âòîðîãî âàðèàíòà: " << performanceSecondVariant << " MFlops" << endl;
    cout << "Ïðîèçâîäèòåëüíîñòü òðåòüåãî âàðèàíòà: " << performanceThirdVariant << " MFlops" << endl;
 
    if (performanceThirdVariant >= 0.3 * performanceSecondVariant) {
        cout << "Òðåòèé âàðèàíò óäîâëåòâîðÿåò óñëîâèþ ïðîèçâîäèòåëüíîñòè." << endl;
    }
    else {
        cout << "Òðåòèé âàðèàíò íå óäîâëåòâîðÿåò óñëîâèþ ïðîèçâîäèòåëüíîñòè." << endl;
    }
 
    return 0;
}

// Команда для компилляции: g++ -O3 -mavx2 -mfma -fopenmp DS_lab2.cpp -lopenblas
