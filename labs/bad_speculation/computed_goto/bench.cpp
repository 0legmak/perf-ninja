#include "solution.h"

#include "benchmark/benchmark.h"

void bench1(benchmark::State& state) {
  const std::string expression = "((((z + u) + (r + q)) + ((a * t) - (l / z))) * (((e - k) / (g + n)) + ((r - n) - (x * k))))";
  const auto bytecode = parse_expression(expression);
  std::vector<int> variable_values(kVarCnt);
  std::fill(variable_values.begin(), variable_values.end(), 1);
  for (auto _ : state) {
    benchmark::DoNotOptimize(interpret_bytecode(bytecode, variable_values));
  }
}

void bench2(benchmark::State& state) {
  const std::string expression = "((((z + u) + (r + q)) + ((a * t) - (l / z))) * (((e - k) / (g + n)) + ((r - n) - (x * k))))";
  const auto bytecode = parse_expression(expression);
  std::vector<int> variable_values(kVarCnt);
  std::fill(variable_values.begin(), variable_values.end(), 1);
  for (auto _ : state) {
    benchmark::DoNotOptimize(interpret_bytecode_opt(bytecode, variable_values));
  }
}

BENCHMARK(bench1)->Iterations(10000000);
BENCHMARK(bench2)->Iterations(10000000);

BENCHMARK_MAIN();
