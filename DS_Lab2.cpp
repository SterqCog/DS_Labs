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
#include <cmath>
#include <random>
#include <intrin.h>
#include "matmul.h"

using namespace std;
using namespace std::chrono;

double calculateComplexity(int n) {
    return 8.0 * pow(n, 3);
}

double calculatePerformance(double complexity, double timeInSeconds) {
    return complexity / (timeInSeconds * 1e6);
}

void multiplyMatricesBLAS(const vector<complex<float>>& A, const vector<complex<float>>& B, vector<complex<float>>& C, const int N) {
    const auto* A_data = reinterpret_cast<const float*>(A.data());
    const auto* B_data = reinterpret_cast<const float*>(B.data());
    auto* C_data = reinterpret_cast<float*>(C.data());

    const float alpha[2] = { 1.0f, 0.0f };
    const float beta[2] = { 0.0f, 0.0f };

    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A_data, N, B_data, N, beta, C_data, N);
}

double calculateFrobeniusNormDiff(int N, const complex<float>* MatrixA, const complex<float>* MatrixB) {
    double sum_sq = 0.0;
#pragma omp parallel for reduction(+:sum_sq) schedule(static)
    for (int i = 0; i < N * N; ++i) {
        float diff_real = MatrixA[i].real() - MatrixB[i].real();
        float diff_imag = MatrixA[i].imag() - MatrixB[i].imag();
        sum_sq += (diff_real * diff_real) + (diff_imag * diff_imag);
    }
    return sqrt(sum_sq);
}

typedef void (*matmul_func_ptr)(int,
    const std::vector<std::complex<float>>&,
    const std::vector<std::complex<float>>&,
    std::vector<std::complex<float>>&);

matmul_func_ptr select_best_matmul() {
    int cpuInfo[4] = { 0 };
    __cpuidex(cpuInfo, 7, 0);

    bool supports_avx512 = (cpuInfo[1] & (1 << 16)) != 0;
    bool supports_avx2 = (cpuInfo[1] & (1 << 5)) != 0;

    if (supports_avx512) {
        cout << "[CPU] Обнаружен AVX-512. Активируем ядро ZMM." << endl;
        return matmul_avx512;
    }
    if (supports_avx2) {
        cout << "[CPU] AVX-512 отсутствует. Активируем ядро AVX2 (YMM)." << endl;
        return matmul_avx2;
    }

    cout << "[CPU] Критическая ошибка: процессор не поддерживает даже AVX2!" << endl;
    return nullptr;
}

int main() {
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);

    static const matmul_func_ptr run_optimized_matmul = select_best_matmul();
    if (run_optimized_matmul == nullptr) {
        return -1;
    }

    int n = 4096;

    vector<complex<float>> A_flat(n * n);
    vector<complex<float>> B_flat(n * n);
    vector<complex<float>> C_blas(n * n);
    vector<complex<float>> C_opt(n * n);

    cout << "\n1. Инициализация матриц данными..." << endl;
    auto init_start = high_resolution_clock::now();

    unsigned int time_seed = static_cast<unsigned int>(std::time(nullptr));
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::mt19937 gen(time_seed + thread_id);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                int idx = i * n + j;
                A_flat[idx] = complex<float>(dis(gen), dis(gen));
                B_flat[idx] = complex<float>(dis(gen), dis(gen));
            }
        }
    }
    auto init_stop = high_resolution_clock::now();
    double init_time = duration_cast<microseconds>(init_stop - init_start).count() / 1e6;
    cout << "-> Инициализация завершена за: " << init_time << " сек.\n" << endl;

    cout << "2. Запуск cblas_cgemm (Вариант 2)..." << endl;
    auto start = high_resolution_clock::now();
    multiplyMatricesBLAS(A_flat, B_flat, C_blas, n);
    auto stop = high_resolution_clock::now();
    double timeSecondVariant = duration_cast<microseconds>(stop - start).count() / 1e6;
    cout << "-> BLAS завершил работу за: " << timeSecondVariant << " сек.\n" << endl;

    cout << "3. Запуск оптимизированного ручного алгоритма (Вариант 3)..." << endl;
    start = high_resolution_clock::now();

    run_optimized_matmul(n, A_flat, B_flat, C_opt);

    stop = high_resolution_clock::now();
    double timeThirdVariant = duration_cast<microseconds>(stop - start).count() / 1e6;
    cout << "-> Ручной алгоритм завершил работу за: " << timeThirdVariant << " сек.\n" << endl;

    cout << "4. Проверка точности вычислений..." << endl;
    double norm_diff = calculateFrobeniusNormDiff(n, C_blas.data(), C_opt.data());
    cout << "-> Норма Фробениуса разности матриц: " << norm_diff << endl;

    double complexity = calculateComplexity(n);
    double performanceSecondVariant = calculatePerformance(complexity, timeSecondVariant);
    double performanceThirdVariant = calculatePerformance(complexity, timeThirdVariant);

    cout << "\n================ ИТОГОВЫЕ РЕЗУЛЬТАТЫ ================" << endl;
    cout << "Время выполнения BLAS:        " << timeSecondVariant << " сек." << endl;
    cout << "Время выполнения ручного кода: " << timeThirdVariant << " сек." << endl;
    cout << "Производительность BLAS:       " << performanceSecondVariant << " MFlops" << endl;
    cout << "Производительность ручного кода: " << performanceThirdVariant << " MFlops" << endl;

    double ratio = (performanceThirdVariant / performanceSecondVariant) * 100.0;
    cout << "Доля ручного алгоритма от BLAS: " << ratio << " %" << endl;

    if (performanceThirdVariant >= 0.3 * performanceSecondVariant) {
        cout << "\nУСПЕХ! Ручной алгоритм выполнил условие (>= 30% от BLAS)." << endl;
    }
    else {
        cout << "\nТРЕБОВАНИЕ НЕ ВЫПОЛНЕНО! Нужно проверить настройки компилятора Release/x64." << endl;
    }

    return 0;
}
