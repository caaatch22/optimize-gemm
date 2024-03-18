#include "matrix.hpp"

Matrix gemm_unrolling(Matrix const& A, Matrix const& B) {
  Matrix C(A.rows(), B.cols());

  for (size_t i = 0; i < A.rows(); i++) {
    for (size_t j = 0; j < B.cols(); j += 8) {
      for (size_t k = 0; k < A.cols(); k++) {
        for (size_t jj = j; jj < j + 8; jj++) {
          C(i, jj) += A(i, k) * B(k, jj);
        }
      }
    }
  }
  return C;
}
