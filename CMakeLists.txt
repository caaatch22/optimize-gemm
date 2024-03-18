cmake_minimum_required(VERSION 3.10)

project(gemm VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

set(SOURCE_FILES
    src/gemm_loopreorder.cpp
    src/gemm_unrolling.cpp
    src/gemm_naive.cpp
    src/gemm_multithreads.cpp
    src/gemm_simd.cpp
    src/gemm_tiling.cpp
    src/gemm_final.cpp
    benchmark.cpp
)

add_executable(benchmark ${SOURCE_FILES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(benchmark PUBLIC OpenMP::OpenMP_CXX)
endif()

target_compile_options(benchmark PRIVATE -O3 -march=native -ffast-math -fopenmp)
