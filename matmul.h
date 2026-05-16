#pragma once
#include <vector>
#include <complex>

void matmul_avx2(int n,
    const std::vector<std::complex<float>>& A,
    const std::vector<std::complex<float>>& B,
    std::vector<std::complex<float>>& C);

void matmul_avx512(int n,
    const std::vector<std::complex<float>>& A,
    const std::vector<std::complex<float>>& B,
    std::vector<std::complex<float>>& C);
