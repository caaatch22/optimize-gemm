import numpy as np
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Matrix multiplication benchmark.')
    parser.add_argument('-m', type=int, default=640, help='First dimension of matrix A and C.')
    parser.add_argument('-k', type=int, default=12800, help='Second dimension of matrix A and first dimension of matrix B.')
    parser.add_argument('-n', type=int, default=640, help='Second dimension of matrix B and C.')
    return parser.parse_args()

args = parse_args()

m, k, n = args.m, args.k, args.n

A = np.random.rand(n, k)
B = np.random.rand(k, m)

start_time = time.time()
C = A @ B
end_time = time.time()

execution_time = end_time - start_time
print(f'A: {m} x {k}, B: {k}, {n}')
gflops = (2 * m * k * n) / (execution_time * 1e9)
print(f"Matrix multiplication executed in {execution_time:.4f} seconds. GFlops: {gflops:.3f}")
