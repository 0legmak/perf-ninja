#include "solution.h"
#include "init.h"
#include "cl_util.h"

#include <iostream>
#include <print>

#include <CL/cl_version.h>
#include <CL/opencl.hpp>

namespace {
  constexpr size_t kVecSize = 100000;
  constexpr auto kMaxError = 1e-5;
} // namespace

int main() {
  try {
    print_devices();
    precompile();
    const auto [a, b] = init(kVecSize);
    const auto ref = reference_solution(a, b);
    const auto sol = solution(a, b, true);
    const auto error = std::fabs((ref - sol) / ref);
    if (error > kMaxError) {
      std::cerr << "Validation Failed." <<
        " Result = " << sol << "."
        " Expected = " << ref << "." <<
        " Error = " << error << "." << std::endl;
      return EXIT_FAILURE;
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
