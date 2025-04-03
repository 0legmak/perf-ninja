#include "solution.h"
#include "init.h"

#include <benchmark/benchmark.h>

namespace {

constexpr size_t kVecSize = 100000;

void bench_ref(benchmark::State& state) {
  const auto [a, b] = init(kVecSize);
  for (auto _ : state) {
    benchmark::DoNotOptimize(reference_solution(a, b));
  }
}

void bench_sol(benchmark::State& state) {
  const auto [a, b] = init(kVecSize);
  precompile();
  for (auto _ : state) {
    benchmark::DoNotOptimize(solution(a, b));
  }
}

}  // namespace

BENCHMARK(bench_ref)->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_sol)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
