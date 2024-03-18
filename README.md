# Optimize GEMM

Several techniques that speedup GEMM(general matrix multiplication), including loop unrolling, loop reorder, tiling, SIMD and multithreads.

Notice that these implementations is correct *if and only if m,n,k(the matrix size) is devisible by 64* except for the *gemm_final* version which is correct for all positive m,n,k.


## Directory Structure
Here is an outline of the main files and directories:
```ccs
.
├── CMakeLists.txt
├── README.md
├── benchmark.cpp
├── gemm_numpy.py
├── include
│   ├── argparse
│   │   └── argparse.hpp
│   ├── fmt
│   │   └── ....h  // not import for this project
│   ├── matrix.hpp
│   └── simd_wrapper.h
└── src
    ├── gemm_final.cpp
    ├── gemm_loopreorder.cpp
    ├── gemm_multithreads.cpp
    ├── gemm_naive.cpp
    ├── gemm_simd.cpp
    ├── gemm_tiling.cpp
    └── gemm_unrolling.cpp

4 directories, 28 files
```

## Build


```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```
This will produce an executable named `benchmark`.


## Runing:
for example, you can only run `tiling` gemm with Matrix A: 640x12800 and Matrix B: 12800 x 6400:

`./build/benchmark --version tiling -m 640 -k 12800 -n 6400`

if you don't specify version, than it will run all algorithms, the matrix size by default is (640x12800), (12800x640). For more usage, run `./benchmark -h`


## Result:

environment:
- Intel(R) Core(TM) i5-1035G1 CPU @ 1.00GHz
- 4 core 8 thread
- avx512
- Linux LAPTOP-QE4VCO5I 5.15.146.1-microsoft-standard-WSL2 

```css
Version: all
Epoches: 1
Matrix dimension: A: 640 x 12800, B: 12800 x 640

gemm_naive: 29.582 s, GFlops: 0.354
gemm_unrolling: 17.281 s, GFlops: 0.607
gemm_loopreorder: 1.137 s, GFlops: 9.225
gemm_tiling: 0.395 s, GFlops: 26.517
gemm_simd: 0.247 s, GFlops: 42.536
gemm_multithreads: 0.077 s, GFlops: 136.884
gemm_final: 0.097 s, GFlops: 108.088
```

For larger matrix, the gemm_final version is better than numpy. 
(10000, 10000) x (10000, 10000)
```
gemm_final: 13.005 s, GFlops: 153.790
gemm_numpy: 16.2833 seconds. GFlops: 122.82
```
