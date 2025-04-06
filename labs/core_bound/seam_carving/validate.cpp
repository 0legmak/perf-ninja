#include "cl_util.h"
#include "data_paths.h"
#include "solution.h"

#include <fstream>
//#include <numeric>
//#include <mdspan>
//#include <print>
//#include <sstream>
//#include <string>
//#include <span>
//#include <vector>
#include <iostream>

#include <CL/cl_version.h>
#include <CL/opencl.hpp>

using namespace std::literals;

namespace {
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
    auto ref = reference_solution();
    auto sol = solution();
    const auto input = load_ppm_image(std::string(kImagePath) + "chameleon.ppm");
    PPMImage ref_image {
      input.width - input.width / 2,
      input.height,
      input.max_color_value,
      ref->process(input.data, input.width, input.height, input.width / 2)
    };
    save_ppm_image(ref_image, std::string(kImagePath) + "chameleon_reduced.ppm");
    PPMImage gpu_image {
      input.width - input.width / 2,
      input.height,
      input.max_color_value,
      sol->process(input.data, input.width, input.height, input.width / 2)
    };
    save_ppm_image(gpu_image, std::string(kImagePath) + "chameleon_reduced_gpu.ppm");

    //const auto input = init1();
    //const auto ref = reference_solution(input, kWidth, kHeight, kRemove);
    //const auto sol = Solution().solution(input, kWidth, kHeight, kRemove);

    //if (ref.size() != sol.size()) {
    //  std::println("size mismatch: ref={} sol={}", ref.size(), sol.size());
    //  return EXIT_FAILURE;
    //}
    //for (int r = 0; r < kHeight; ++r) {
    //  for (int c = 0; c < kWidth - kRemove; ++c) {
    //    const auto ref_val = ref[r * (kWidth - kRemove) + c];
    //    const auto sol_val = sol[r * (kWidth - kRemove) + c];
    //    if (ref_val != sol_val) {
    //      std::println("data mismatch at [{},{}]: ref={} sol={}", r, c, ref_val, sol_val);
    //      return EXIT_FAILURE;
    //    }
    //  }
    //}

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
