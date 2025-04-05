#include "solution.h"
#include "init.h"
#include "cl_util.h"

#include <iostream>

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
    const auto ref = reference_solution();
    ref->set_input(a, b, N, K, M);
    ref->run_kernel();
    const auto ref_res = ref->get_output();
    const auto sol = solution();
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
    std::cerr << "OpenCL build error: " << err.what() << ", code: " << get_error_string(err.err()) << '\n';
    for (const auto& [_, log] : err.getBuildLog()) {
      std::cerr << log << '\n';
    }
  } catch (const cl::Error& err) {
    std::cerr << "OpenCL error: " << err.what() << ", code: " << get_error_string(err.err()) << '\n';
  } catch (const std::exception& err) {
    std::cerr << "C++ exception: " << err.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error" << '\n';
  }
  return EXIT_FAILURE;
}
