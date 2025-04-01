#include <fstream>
#include <numeric>
#include <mdspan>
#include <print>
#include <sstream>
#include <string>
#include <span>
#include <vector>
#include <CL/cl_version.h>
#include <CL/opencl.hpp>
#include "data_paths.h"
#include "solution.h"

using namespace std::literals;

namespace {
  void print_devices() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::println("Available OpenCL platforms and devices:");
    for (const auto& platform : platforms)	{
      std::println("\tPlatform {}; version: {}; vendor: {}",
        platform.getInfo<CL_PLATFORM_NAME>(),
        platform.getInfo<CL_PLATFORM_VERSION>(),
        platform.getInfo<CL_PLATFORM_VENDOR>()
      );
      std::vector<cl::Device> devices;
      platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
      for (const auto& device : devices) {
        auto get_device_type = [&]() {
          const auto device_type = device.getInfo<CL_DEVICE_TYPE>();
          constexpr int device_count = 4;
          constexpr std::array<std::pair<int, std::string_view>, device_count> types = {{
            { CL_DEVICE_TYPE_CPU, "CPU"sv },
            { CL_DEVICE_TYPE_GPU, "GPU"sv },
            { CL_DEVICE_TYPE_ACCELERATOR, "ACCELERATOR"sv }
          }};
          for (const auto [type, label] : types) {
            if (device_type & type) {
              return label;
            }
          }
          return "UNKNOWN"sv;
        };
        std::println(
          "\t\t{} {}\n\t\t\tVersion: {}\n\t\t\tVendor: {}\n\t\t\tCompute units: {}\n\t\t\t"
          "Max work group size: {}\n\t\t\tMax work item dimensions: {}\n\t\t\tMax work item sizes: {}\n\t\t\t"
          "Max memory size: {}\n\t\t\tMax allocatable memory: {}",
          get_device_type(),
          device.getInfo<CL_DEVICE_NAME>(),
          device.getInfo<CL_DEVICE_VERSION>(),
          device.getInfo<CL_DEVICE_VENDOR>(),
          device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(),
          device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(),
          device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>(),
          device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>(),
          device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>(),
          device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()
        );
      }
    }
  }

} // namespace

int main() {
  try {
    print_devices();

    const auto input = init1();
    const auto ref = reference_solution(input, kWidth, kHeight);
    const auto sol = Solution().solution(input, kWidth, kHeight);

    if (ref.size() != sol.size()) {
      std::println("size mismatch: ref={} sol={}", ref.size(), sol.size());
      return EXIT_FAILURE;
    }
    for (int r = 0; r < kHeight - 2; ++r) {
      for (int c = 0; c < kWidth - 2; ++c) {
        const auto ref_val = ref[r * (kWidth - 2) + c];
        const auto sol_val = sol[r * (kWidth - 2) + c];
        constexpr float kError = 1e-6;
        const auto error = fabs(ref_val - sol_val) / ref_val;
        if (error > kError) {
          std::println("data mismatch at [{},{}]: ref={} sol={} error={}", r, c, ref_val, sol_val, error);
          return EXIT_FAILURE;
        }
      }
    }
    std::println("Validation Successful");

    //for (int r = 0; r < kHeight; ++r) {
    //  for (int c = 0; c < kWidth; ++c) {
    //    std::print("{} ", input[r * kWidth + c]);
    //  }
    //  std::println("");
    //}
    //std::println("");
    //for (int r = 0; r < kHeight - 2; ++r) {
    //  for (int c = 0; c < kWidth - 2; ++c) {
    //    std::print("{} ", ref[r * (kWidth - 2) + c]);
    //  }
    //  std::println("");
    //}
    //std::println("");
    //for (int r = 0; r < kHeight - 2; ++r) {
    //  for (int c = 0; c < kWidth - 2; ++c) {
    //    std::print("{} ", sol[r * (kWidth - 2) + c]);
    //  }
    //  std::println("");
    //}


  //  cl::Context cl_context = cl::Context::getDefault();
  //  for (const auto& device : cl_context.getInfo<CL_CONTEXT_DEVICES>()) {
  //    std::println("Using device: {}", device.getInfo<CL_DEVICE_NAME>());
  //  }
  //  cl::CommandQueue cl_command_queue(cl_context);

  //  const auto kernel_sources = load_file(kernels_cl_path);
  //  cl::Program cl_program(cl_context, kernel_sources, /* build */ true);

  //  constexpr auto vector_size = 1000;
    //std::vector<int> a(vector_size); // = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  //  std::iota(a.begin(), a.end(), 0);
    //std::vector<int> b(vector_size); // = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };
  //  std::iota(b.begin(), b.end(), 0);
    //const auto vector_size_bytes = vector_size * sizeof(a[0]);

    //cl::Buffer buffer_a(cl_context, a.begin(), a.end(), true);
    //cl::Buffer buffer_b(cl_context, b.begin(), b.end(), true);
  //  cl::Buffer buffer_c(cl_context, CL_MEM_READ_WRITE, vector_size_bytes);

  //  auto vector_add_kernel = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(cl_program, "vector_add");
  //  vector_add_kernel(
  //    cl::EnqueueArgs(
  //      cl_command_queue,
  //      cl::NDRange(vector_size)
  //    ),
  //    buffer_a,
  //    buffer_b,
  //    buffer_c
  //  );

  //  std::vector<int> c(vector_size);
    //cl_command_queue.enqueueReadBuffer(buffer_c, CL_TRUE, 0, vector_size_bytes, c.data());
  //  std::println("{}", std::span(c.begin() + c.size() - 10, c.end()));

  } catch (const cl::BuildError& err) {
    std::println("OpenCL build error: {}, code: {}", err.what(), err.err());
    for (const auto& [_, log] : err.getBuildLog()) {
      std::println("{}", log);
    }
  } catch (const cl::Error& err) {
    std::println("OpenCL error: {}, code: {}", err.what(), err.err());
  } catch (const std::exception& err) {
    std::println("C++ exception: {}", err.what());
  } catch (...) {
    std::println("Unknown error");
  }
  return EXIT_SUCCESS;
}
