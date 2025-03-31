#include "solution.h"
#include "data_paths.h"
 
#include <fstream>
#include <numeric>
#include <mdspan>
#include <print>
#include <sstream>
#include <string>
#include <span>

namespace {
  std::string load_file(const std::string& file_path) {
    std::ifstream fileStream(file_path);
    std::stringstream buffer;
    buffer << fileStream.rdbuf();
    return buffer.str();
  }
}

std::vector<float> reference_solution(const std::vector<RGB>& input, int width, int height) {
  const std::mdspan<const RGB, std::dextents<size_t, 2>> pixels(input.data(), height, width);
  auto squared_diff = [](float v1, float v2) {
    return (v1 - v2) * (v1 - v2);
  };
  auto squared_gradient = [&squared_diff](const RGB c1, const RGB c2) {
    return squared_diff(c1[0], c2[0]) + squared_diff(c1[1], c2[1]) + squared_diff(c1[2], c2[2]);
  };
  auto calc_energy = [&pixels, &squared_gradient](int row, int col) {
    return std::sqrt(
      squared_gradient(pixels[row, col - 1], pixels[row, col + 1]) +
      squared_gradient(pixels[row - 1, col], pixels[row + 1, col])
    );
  };
  const int energy_width = width - 2;
  const int energy_height = height - 2;
  std::vector<float> energy_buffer(energy_width * energy_height);
  const std::mdspan<float, std::dextents<size_t, 2>> energy(energy_buffer.data(), energy_height, energy_width);
  for (int row = 0; row < energy_height; ++row) {
    for (int col = 0; col < energy_width; ++col) {
      energy[row, col] = calc_energy(row + 1, col + 1);
      //energy[row, col] = pixels[row, col][0];
    }
  }
  //constexpr float kBorderEnergy = 1000.0f;
  //for (int row = 0; row < height; ++row) {
  //  energy[row][0] = energy[row][width - 1] = kBorderEnergy;
  //}
  //for (int col = 0; col < width; ++col) {
  //  energy[0][col] = energy[height - 1][col] = kBorderEnergy;
  //}
  return energy_buffer;
}

Solution::Solution() {
  cl_context = cl::Context::getDefault();
  //for (const auto& device : cl_context.getInfo<CL_CONTEXT_DEVICES>()) {
  //  std::println("Using device: {}", device.getInfo<CL_DEVICE_NAME>());
  //}
  cl_command_queue = cl::CommandQueue(cl_context);
  const auto kernel_sources = load_file(kernels_cl_path);
  cl_program = cl::Program(cl_context, kernel_sources, /* build */ true);
  calc_energy_kernel = std::make_unique<cl::KernelFunctor<cl::Buffer, int, cl::Buffer>>(cl_program, "vector_calc_energy");
}

std::vector<float> Solution::solution(const std::vector<RGB>& input, int width, int height) {
	cl::Buffer pixel_buffer(cl_context, input.begin(), input.end(), true);
  const int output_size = (width - 2) * (height - 2);
  const int output_size_bytes = output_size * sizeof(float);
  cl::Buffer output_buffer(cl_context, CL_MEM_WRITE_ONLY, output_size_bytes);
  (*calc_energy_kernel)(
    cl::EnqueueArgs(
      cl_command_queue,
      cl::NDRange(height - 2, width - 2)
    ),
    pixel_buffer,
    width,
    output_buffer
  );
  std::vector<float> output(output_size);
	cl_command_queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output_size_bytes, output.data());
  return output;
}
