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

  struct PPMImage {
    int width;
    int height;
    int max_color_value;
    std::vector<RGB> data;
  };

  PPMImage load_ppm_image(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }
    std::string magic_number;
    file >> magic_number;
    if (magic_number != "P6") {
        throw std::runtime_error("Invalid PPM file: " + file_path);
    }
    int width, height, max_color_value;
    file >> width >> height >> max_color_value;
    file.ignore(1); // Skip the newline character after the header
    std::vector<RGB> data(width * height);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(RGB));
    if (!file) {
        throw std::runtime_error("Failed to read image data from file: " + file_path);
    }
    return {width, height, max_color_value, data};
  }

  void save_ppm_image(const PPMImage& image, const std::string& file_path) {
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + file_path);
    }
    file << "P6\n";
    file << image.width << ' ' << image.height << '\n';
    file << image.max_color_value << '\n';
    file.write(reinterpret_cast<const char*>(image.data.data()), image.data.size() * sizeof(RGB));
    if (!file) {
      throw std::runtime_error("Failed to write image data to file: " + file_path);
    }
  }

} // namespace

int main() {
  try {
    print_devices();
    auto image = load_ppm_image("C:\\Users\\admin\\source\\repos\\perf-ninja-fork\\labs\\core_bound\\seam_carving\\chameleon.ppm");
    image.data = reference_solution(image.data, image.width, image.height, image.width / 2);
    image.width /= 2;
    save_ppm_image(image, "C:\\Users\\admin\\source\\repos\\perf-ninja-fork\\labs\\core_bound\\seam_carving\\chameleon_reduced.ppm");

    //const auto input = init1();
    //const auto ref = reference_solution(input, kWidth, kHeight);
    //const auto sol = Solution().solution(input, kWidth, kHeight);

    //if (ref.size() != sol.size()) {
    //  std::println("size mismatch: ref={} sol={}", ref.size(), sol.size());
    //  return EXIT_FAILURE;
    //}
    //for (int r = 0; r < kHeight - 2; ++r) {
    //  for (int c = 0; c < kWidth - 2; ++c) {
    //    const auto ref_val = ref[r * (kWidth - 2) + c];
    //    const auto sol_val = sol[r * (kWidth - 2) + c];
    //    constexpr float kError = 1e-6;
    //    const auto error = fabs(ref_val - sol_val) / ref_val;
    //    if (error > kError) {
    //      std::println("data mismatch at [{},{}]: ref={} sol={} error={}", r, c, ref_val, sol_val, error);
    //      return EXIT_FAILURE;
    //    }
    //  }
    //}
    //std::println("Validation Successful");

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
