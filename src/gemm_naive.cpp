#include "matrix.hpp"

Matrix gemm_naive(Matrix const& A, Matrix const& B) {
  Matrix C(A.rows(), B.cols());
  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = 0; j < B.cols(); j++) {
      for (size_t k = 0; k < A.cols(); k++) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
  return C;
}