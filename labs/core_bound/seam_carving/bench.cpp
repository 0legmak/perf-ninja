#include "solution.h"

#include <benchmark/benchmark.h>

namespace {

void bench_ref(benchmark::State& state) {
  const auto input = init1();
  for (auto _ : state) {
    benchmark::DoNotOptimize(reference_solution(input, kWidth, kHeight));
  }
}

void bench_sol(benchmark::State& state) {
  const auto input = init1();
  Solution sol;
  for (auto _ : state) {
    benchmark::DoNotOptimize(sol.solution(input, kWidth, kHeight));
  }
}

}  // namespace

BENCHMARK(bench_ref)->Unit(benchmark::kMillisecond);
BENCHMARK(bench_sol)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
