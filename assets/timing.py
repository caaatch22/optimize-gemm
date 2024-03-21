import subprocess
import re
import matplotlib.pyplot as plt

versions = ['naive', 'unrolling', 'loopreorder', 'tiling', 
            'simd', 'multithreads', 'final', 'numpy']

matrix_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
gflops_data = {version: [] for version in versions}


def run(cmd: str, size: int, version: str):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    matches = re.search(r"GFlops: (\d+\.\d+)", result.stdout)
    if matches:
        gflops = float(matches.group(1))
        gflops_data[version].append(gflops)
        print(f"Version: {version}, Matrix Size: {size}, GFlops: {gflops}")
    else:
        gflops_data[version].append(0)
        print(f"Version: {version}, Matrix Size: {size}, GFlops data not found.")
        

for size in matrix_sizes:
    for version in versions:
        if version == 'numpy':
            cmd = f"python3 ../gemm_numpy.py -m {size} -k {size} -n {size}"
            run(cmd, size, version)
            continue
        # Skip this as it takes too long
        if version in ['naive', 'unrolling'] and size > 2048:
            continue
        
        cmd = f"../build/benchmark --version {version} -m {size} -k {size} -n {size}"
        run(cmd, size, version)


plt.figure(figsize=(10, 6))
for version, gflops in gflops_data.items():
    if version in ["naive", "unrolling"]:
        plt.plot(matrix_sizes[:-2], gflops, label=version)
    else:
        plt.plot(matrix_sizes, gflops, label=version)

    last_size = matrix_sizes[-1] if version not in ["naive", "unrolling"] else matrix_sizes[-3]
    last_gflops = gflops[-1]
    plt.text(last_size, last_gflops, f"{last_gflops:.2f}", ha='center', va='bottom')

plt.xlabel("Matrix Size")
plt.ylabel("GFlops")
plt.title("GEMM Performance")
plt.legend()
plt.grid(True)
plt.xscale('log', base=2)
plt.yscale('log', base=10)
plt.xticks(matrix_sizes, labels=[str(size) for size in matrix_sizes])
plt.savefig("gemm_performance.png")
