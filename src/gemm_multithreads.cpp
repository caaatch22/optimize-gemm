#include <omp.h>

#include "fmt/format.h"
#include "matrix.hpp"
#include "simd_wrapper.h"

Matrix gemm_threads(Matrix const& A, Matrix const& B) {
  static constexpr size_t BLOCK_SIZE = SIMD_SIZE * 8;
  alignas(BLOCK_SIZE) static float local_a[BLOCK_SIZE][BLOCK_SIZE];
  alignas(BLOCK_SIZE) static float local_b[BLOCK_SIZE][BLOCK_SIZE];
  alignas(BLOCK_SIZE) static float local_c[BLOCK_SIZE][BLOCK_SIZE];

#pragma omp threadprivate(local_a, local_b, local_c)

  Matrix C = Matrix(A.rows(), B.cols());

  size_t const ai_block_num = A.rows() / BLOCK_SIZE;
  size_t const aj_block_num = B.cols() / BLOCK_SIZE;
  size_t const bk_block_num = B.rows() / BLOCK_SIZE;
  fmt::print("ai_block_num: {}, aj_block_num: {}, bk_block_num: {}\n",
             ai_block_num, aj_block_num, bk_block_num);

#pragma omp parallel for
  for (size_t bi = 0; bi < ai_block_num; bi++) {
    for (size_t bj = 0; bj < aj_block_num; bj++) {
      // Clear localC.
      for (size_t i = 0; i < BLOCK_SIZE; i++) {
        for (size_t j = 0; j < BLOCK_SIZE; j++) {
          local_c[i][j] = 0.f;
        }
      }

      for (size_t bk = 0; bk < bk_block_num; bk++) {
        // Copy local block.
        for (size_t i = 0; i < BLOCK_SIZE; i++) {
          for (size_t j = 0; j < BLOCK_SIZE; j++) {
            size_t const ax = bi * BLOCK_SIZE + i;
            size_t const ay = bk * BLOCK_SIZE + j;
            size_t const bx = bk * BLOCK_SIZE + i;
            size_t const by = bj * BLOCK_SIZE + j;
            local_a[i][j] = A(ax, ay);
            local_b[i][j] = B(bx, by);
          }
        }

        // BLOCK_GEMM
        for (size_t i = 0; i < BLOCK_SIZE; i++) {
          for (size_t k = 0; k < BLOCK_SIZE; k++) {
            auto a = SIMD_SET1(local_a[i][k]);
            for (size_t j = 0; j < BLOCK_SIZE; j += SIMD_SIZE) {
              auto b = SIMD_LOAD(&local_b[k][j]);
              auto c = SIMD_LOAD(&local_c[i][j]);
              c = SIMD_FMADD(a, b, c);
              SIMD_STORE(&local_c[i][j], c);
            }
          }
        }
      }

      for (size_t i = 0; i < BLOCK_SIZE; i++) {
        for (size_t j = 0; j < BLOCK_SIZE; j++) {
          C(bi * BLOCK_SIZE + i, bj * BLOCK_SIZE + j) = local_c[i][j];
        }
      }
    }
  }

  return C;
}