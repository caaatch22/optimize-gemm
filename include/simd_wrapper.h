#ifndef SIMD_WRAPPER_H
#define SIMD_WRAPPER_H

#if defined(__AVX512F__)
#include <immintrin.h>

#define SIMD_LOAD _mm512_load_ps
#define SIMD_SET1 _mm512_set1_ps
#define SIMD_FMADD _mm512_fmadd_ps
#define SIMD_STORE _mm512_store_ps
#define SIMD_SIZE 16  // 512 / 32
#define SIMD_VERSION "AVX512"

// 检查 AVX2 支持
#elif (defined(__AVX2__))
#include <immintrin.h>

#define SIMD_LOAD _mm256_load_ps
#define SIMD_SET1 _mm256_set1_ps
#define SIMD_FMADD _mm256_fmadd_ps  
#define SIMD_STORE _mm256_store_ps
#define SIMD_SIZE 8
#define SIMD_VERSION "AVX2"

#elif defined(__AVX__)
#include <immintrin.h>

#define SIMD_LOAD _mm256_load_ps
#define SIMD_SET1 _mm256_set1_ps
#define SIMD_FMADD(a, b, c) _mm256_add_ps(_mm256_mul_ps(a, b), c)
#define SIMD_STORE _mm256_store_ps
#define SIMD_SIZE 8
#define SIMD_VERSION "AVX"


#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

#define SIMD_LOAD vld1q_f32
#define SIMD_SET1 vdupq_n_f32
#define SIMD_FMADD(a, b, c) vmlaq_f32(c, a, b)
#define SIMD_STORE vst1q_f32
#define SIMD_SIZE 4
#define SIMD_VERSION "NEON"

// 检查 SSE4.2 支持
#elif defined(__SSE4_2__) || defined(__SSE4_1__)
#include <immintrin.h>

#define SIMD_LOAD _mm_load_ps
#define SIMD_SET1 _mm_set1_ps
#define SIMD_FMADD(a, b, c) _mm_add_ps(_mm_mul_ps(a, b), c)
#define SIMD_STORE _mm_store_ps
#define SIMD_SIZE 4
#define SIMD_VERSION "SEE4"

#else
#error "No supported SIMD instructions set found."
#endif

#endif  // SIMD_WRAPPER
