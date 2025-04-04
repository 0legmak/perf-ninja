#include "solution.h"
#include "init.h"

#include <benchmark/benchmark.h>

namespace {

constexpr size_t N = 501;
constexpr size_t K = 602;
constexpr size_t M = 703;

void bench_ref(benchmark::State& state) {
  const auto [a, b] = init(N, K, M);
  const auto ref = reference_solution();
  ref->set_input(a, b, N, K, M);
  for (auto _ : state) {
    ref->run_kernel();
  }
}

void bench_sol(benchmark::State& state) {
  const auto [a, b] = init(N, K, M);
  const auto sol = solution();
  sol->set_input(a, b, N, K, M);
  for (auto _ : state) {
    sol->run_kernel();
  }
}

}  // namespace

BENCHMARK(bench_ref)->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_sol)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
