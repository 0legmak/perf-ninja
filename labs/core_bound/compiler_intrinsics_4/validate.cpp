#include "data_paths.h"
#include "picture.h"
#include "solution.h"
#include "wait_for_debugger.h"

#include <print>
#include <fstream>
#include <chrono>

int main() {
  const auto start1 = std::chrono::high_resolution_clock::now();
  const auto data = mandelbrot();
  const auto finish1 = std::chrono::high_resolution_clock::now();
  std::println("{}", std::chrono::duration_cast<std::chrono::milliseconds>(finish1 - start1));
  const auto start = std::chrono::high_resolution_clock::now();
  const auto image = generate_ppm_image(data);
  const auto finish = std::chrono::high_resolution_clock::now();
  std::println("{}", std::chrono::duration_cast<std::chrono::milliseconds>(finish - start));
  std::ofstream(output_image_path) << image;
  return 0;
}
