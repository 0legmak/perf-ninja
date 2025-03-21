#include "solution.h"
#include "thread_pool.h"
#include "const.h"

#include "benchmark/benchmark.h"

namespace {

void bench1(benchmark::State& state) {
  get_thread_pool();
  std::vector<short> data;
  for (auto _ : state) {
    data = mandelbrot(kImageWidth, kImageHeight, (ImplType)state.range(0));
    benchmark::DoNotOptimize(data);
  }
}

}  // namespace

BENCHMARK(bench1)->
  Unit(benchmark::kMillisecond)->
  Arg(ImplType::kOriginal)->
  Arg(ImplType::kVectorized)->
  // Arg(ImplType::kThreadPool)->
  Arg(ImplType::kOpenMP);
BENCHMARK_MAIN();
