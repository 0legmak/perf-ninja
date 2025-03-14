#include "benchmark/benchmark.h"

namespace {
  void bench(benchmark::State& state) {
    std::string x = "hello";
    for (auto _ : state)
      std::string copy(x);
  }
}

BENCHMARK(bench);

BENCHMARK_MAIN();
