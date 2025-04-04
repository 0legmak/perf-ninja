#include "solution.h"
#include "init.h"
#include "cl_util.h"

#include <iostream>
#include <print>

#include <CL/cl_version.h>
#include <CL/opencl.hpp>

namespace {
  constexpr size_t N = 111;
  constexpr size_t K = 222;
  constexpr size_t M = 333;
  constexpr auto kMaxError = 1e-5;
} // namespace

int main() {
  try {
    print_devices();
    cl::CommandQueue::setDefault(cl::CommandQueue(cl::QueueProperties::Profiling));
    const auto [a, b] = init(N, K, M);
    const auto ref = reference_solution(true);
    ref->set_input(a, b, N, K, M);
    ref->run_kernel();
    const auto ref_res = ref->get_output();
    const auto sol = solution(true);
    sol->set_input(a, b, N, K, M);
    sol->run_kernel();
    const auto res = sol->get_output();
    if (res.size() != ref_res.size()) {
      std::cerr << "Validation Failed." <<
        " Result size = " << res.size() << "."
        " Expected size = " << ref_res.size() << "." << std::endl;
      return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < M; ++j) {
        const auto error = std::fabs((ref_res[i * M + j] - res[i * M + j]) / ref_res[i * M + j]);
        if (error > kMaxError) {
          std::cerr << "Validation Failed." <<
            " Result[" << i << ", " << j << "] = " << res[i * M + j] << "."
            " Expected = " << ref_res[i * M + j] << "." <<
            " Error = " << error << "." << std::endl;
          return EXIT_FAILURE;
        }
      }
    }
    std::cout << "Validation Successful" << std::endl;
    return EXIT_SUCCESS;
  } catch (const cl::BuildError& err) {
    std::println(std::cerr, "OpenCL build error: {}, code: {}", err.what(), get_error_string(err.err()));
    for (const auto& [_, log] : err.getBuildLog()) {
      std::println("{}", log);
    }
  } catch (const cl::Error& err) {
    std::println("OpenCL error: {}, code: {}", err.what(), get_error_string(err.err()));
  } catch (const std::exception& err) {
    std::println("C++ exception: {}", err.what());
  } catch (...) {
    std::println("Unknown error");
  }
  return EXIT_FAILURE;
}
