#include "solution.h"

#include <benchmark/benchmark.h>

namespace {

void bench_ref(benchmark::State& state) {
  const auto input = init1();
  auto sol = reference_solution();
  for (auto _ : state) {
    benchmark::DoNotOptimize(sol->process(input, kWidth, kHeight, kRemove));
  }
}

void bench_sol(benchmark::State& state) {
  const auto input = init1();
  auto sol = solution();
  for (auto _ : state) {
    benchmark::DoNotOptimize(sol->process(input, kWidth, kHeight, kRemove));
  }
}

}  // namespace

BENCHMARK(bench_ref)->Unit(benchmark::kMillisecond);
BENCHMARK(bench_sol)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
