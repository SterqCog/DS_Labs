// matmul.h
#pragma once
#include <vector>
#include <complex>

// Прототип функции для AVX2
void matmul_avx2(int n,
    const std::vector<std::complex<float>>& A,
    const std::vector<std::complex<float>>& B,
    std::vector<std::complex<float>>& C);

// Прототип функции для AVX-512
void matmul_avx512(int n,
    const std::vector<std::complex<float>>& A,
    const std::vector<std::complex<float>>& B,
    std::vector<std::complex<float>>& C);