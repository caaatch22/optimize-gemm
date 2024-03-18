#include "fmt/format.h"
#include "matrix.hpp"

Matrix gemm_tiling(Matrix const& A, Matrix const& B) {
  static constexpr size_t BLOCK_SIZE = 64;

  Matrix C = Matrix(A.rows(), B.cols());

  size_t const ai_block_num = A.rows() / BLOCK_SIZE;
  size_t const aj_block_num = B.cols() / BLOCK_SIZE;
  size_t const bk_block_num = B.rows() / BLOCK_SIZE;
  fmt::print("ai_block_num: {}, aj_block_num: {}, bk_block_num: {}\n",
             ai_block_num, aj_block_num, bk_block_num);

  for (size_t bi = 0; bi < ai_block_num; bi++) {
    for (size_t bj = 0; bj < aj_block_num; bj++) {
      for (size_t bk = 0; bk < bk_block_num; bk++) {
        for (size_t i = 0; i < BLOCK_SIZE; i++) {
          for (size_t k = 0; k < BLOCK_SIZE; k++) {
            for (size_t j = 0; j < BLOCK_SIZE; j++) {
              size_t const ax = bi * BLOCK_SIZE + i;
              size_t const ay = bk * BLOCK_SIZE + k;
              size_t const bx = bk * BLOCK_SIZE + k;
              size_t const by = bj * BLOCK_SIZE + j;
              C(ax, by) += A(ax, ay) * B(bx, by);
            }
          }
        }
      }
    }
  }
  return C;
}
