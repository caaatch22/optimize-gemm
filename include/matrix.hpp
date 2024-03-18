#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <ranges>
#include <vector>

class [[nodiscard]] Timer {
 public:
  Timer() noexcept : start(std::chrono::high_resolution_clock::now()) {}
  [[nodiscard]] auto elapsed() const noexcept -> std::chrono::duration<double> {
    const auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - start);
  }
  void reset() noexcept { start = std::chrono::high_resolution_clock::now(); }
 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

static std::random_device seed_random_device;
static std::mt19937 rand_engine(seed_random_device());

inline static float rand_float() {
  return static_cast<float>(rand_engine()) / rand_engine.max();
}

class [[nodiscard]] Matrix {
 public:
  static constexpr size_t ALIGNMENT = 64;
  Matrix(size_t m, size_t n) : m_(m), n_(n), data_(new(std::align_val_t(ALIGNMENT)) float[m * n]) {}
  ~Matrix() { delete[] data_; }
  Matrix(Matrix const& rhs) : m_(rhs.m_), n_(rhs.n_), data_(new(std::align_val_t(ALIGNMENT)) float[m_ * n_]) {
    std::copy(rhs.data_, rhs.data_ + m_ * n_, data_);
  }
  Matrix(Matrix&& rhs) noexcept = default;
  Matrix& operator=(Matrix&& rhs) noexcept = default;
  Matrix& operator=(Matrix const& rhs) {
    if (this != std::addressof(rhs)) {
        Matrix tmp(rhs);
        swap(tmp, *this);
    }
    return *this;
  }

  friend void swap(Matrix& lhs, Matrix& rhs) noexcept {
    using std::swap;
    swap(lhs.m_, rhs.m_);
    swap(lhs.n_, rhs.n_);
    swap(lhs.data_, rhs.data_);
  }

  [[nodiscard]] bool operator==(Matrix const& rhs) const noexcept {
    static const float EPS = 1e-6;
    return m_ == rhs.m_ and n_ == rhs.n_ and
           std::equal(data_, data_ + size(), rhs.data_,
                      [](float a, float b) { return std::abs(a - b) < EPS; });
  }
  [[nodiscard]] bool operator!=(Matrix const& rhs) const noexcept { return !(*this == rhs); }
  [[nodiscard]] size_t rows() const noexcept { return m_; }
  [[nodiscard]] size_t cols() const noexcept { return n_; }
  [[nodiscard]] size_t size() const noexcept { return m_ * n_; }

  void zeros() { std::fill(data_, data_ + size(), 0.f); }
  void ones() { std::fill(data_, data_ + size(), 1.f); }
  void rand() { std::generate(data_, data_ + size(), rand_float); }

  [[nodiscard]] float& operator()(size_t i, size_t j) { return data_[i * n_ + j]; }
  [[nodiscard]] float const& operator()(size_t i, size_t j) const { return data_[i * n_ + j];}

  Matrix make_aligned(size_t const block_size) const {
    size_t const new_row = ((rows() + block_size - 1) / block_size) * block_size;
    size_t const new_col = ((cols() + block_size - 1) / block_size) * block_size;
    Matrix result(new_row, new_col);
    for (size_t i = 0; i < rows(); i++) {
      for (size_t j = 0; j < cols(); j++) {
        result(i, j) = (*this)(i, j);
      }
    }
    return result;
  }

  void from_aligned(Matrix const& src) {
    for (size_t i = 0; i < rows(); i++) {
      for (size_t j = 0; j < cols(); j++) {
        (*this)(i, j) = src(i, j);
      }
    }
  }

private:
  size_t m_;
  size_t n_;
  alignas(64) float* data_;
};

inline double GFlops(double sec, size_t m, size_t n, size_t k) {
  return 2.0 * m * n * k / sec / 1e9;
}

Matrix gemm_naive(Matrix const& A, Matrix const& B);
Matrix gemm_unrolling(Matrix const& A, Matrix const& B);
Matrix gemm_loopreorder(Matrix const& A, Matrix const& B);
Matrix gemm_tiling(Matrix const& A, Matrix const& B);
Matrix gemm_simd(Matrix const& A, Matrix const& B);
Matrix gemm_threads(Matrix const& A, Matrix const& B);
Matrix gemm_final(Matrix const& A, Matrix const& B);


#endif  // MATRIX_H
