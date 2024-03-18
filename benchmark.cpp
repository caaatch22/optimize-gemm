#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#define FMT_HEADER_ONLY
#include "argparse/argparse.hpp"
#include "fmt/format.h"
#include "matrix.hpp"

using GemmFunc = std::function<Matrix(Matrix const&, Matrix const&)>;

const std::unordered_map<std::string, GemmFunc> versions{
    {"none", {}},
    {"naive", gemm_naive},
    {"unrolling", gemm_unrolling},
    {"loopreorder", gemm_loopreorder},
    {"tiling", gemm_tiling},
    {"simd", gemm_simd},
    {"multithreads", gemm_threads},
    {"final", gemm_final},
    {"all", {}},
};

struct Args {
  std::string version;
  size_t epoches;
  size_t m;
  size_t k;
  size_t n;
};

Args parseArgs(int argc, char* argv[]) {
  argparse::ArgumentParser program("gemm_benchmark");

  program.add_argument("--version")
      .default_value(std::string("all"))
      .help(
          "Version to use: naive, unrolling, loopreorder, tiling, simd, "
          "multithreads, or run gemm with all. Default is all.")
      .action([&](std::string const& value) {
        if (versions.find(value) == versions.end()) {
          throw std::runtime_error("Invalid version: " + value);
        }
        return value;
      });

  program.add_argument("--epoches")
      .scan<'i', int>()
      .default_value(1)
      .help("Number of epochs. Accepts an integer. Default is 1.");

  auto add_dim_args = [&](std::string const& name, int default_value,
                          std::string const& help) {
    program.add_argument(name)
        .scan<'i', int>()
        .default_value(default_value)
        .help(help);
  };
  add_dim_args("-m", 640, "First dimension of the matrix A(M x K).");
  add_dim_args("-k", 12800, "Second dimension of the matrix A(M x K).");
  add_dim_args("-n", 640, "Second dimension of the matrix B(K x N).");

  try {
    program.parse_args(argc, argv);
  } catch (std::runtime_error const& err) {
    fmt::print(stderr, "{}\n", err.what());
    std::cerr << program;
    exit(1);
  }
  Args ret;
  ret.version = program.get<std::string>("--version");
  ret.epoches = program.get<int>("--epoches");
  ret.m = program.get<int>("-m");
  ret.k = program.get<int>("-k");
  ret.n = program.get<int>("-n");
  return ret;
}

Matrix run(Matrix const& A, Matrix const& B, std::string const& version) {
  fmt::print("gemm_{}:\n", version);
  Timer timer;
  auto C = versions.at(version)(A, B);
  double const time = timer.elapsed().count();

  fmt::print("{:.3f} s, GFlops: {:.3f}\n", time,
             GFlops(time, A.rows(), A.cols(), B.cols()));
  fmt::print("\n");
  return C;
}

int main(int argc, char* argv[]) {
  Args args = parseArgs(argc, argv);

  fmt::print("Version: {}\n", args.version);
  fmt::print("Epoches: {}\n", args.epoches);
  fmt::print("Matrix dimension: A: {} x {}, B: {} x {}\n", args.m, args.k,
             args.k, args.n);
  fmt::print("\n");

  Matrix A(args.m, args.k);
  Matrix B(args.k, args.n);
  A.rand();
  B.rand();

  if (args.version == "none") {
    return 0;
  }

  if (args.version == "all") {
    for (size_t i = 0; i < args.epoches; i++) {
      auto const C0 = run(A, B, "naive");
      auto const C1 = run(A, B, "unrolling");
      auto const C2 = run(A, B, "loopreorder");
      auto const C3 = run(A, B, "tiling");
      auto const C4 = run(A, B, "simd");
      auto const C5 = run(A, B, "multithreads");
      auto const C6 = run(A, B, "final");
    }
    return 0;
  }

  for (size_t i = 0; i < args.epoches; i++) {
    auto const C = run(A, B, args.version);
  }

  return 0;
}
