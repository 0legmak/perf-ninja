#include "data_paths.h"
#include "picture.h"
#include "solution.h"
#include "wait_for_debugger.h"
#include "thread_pool.h"

#include <chrono>
#include <ios>
#include <iostream>
#include <fstream>

int main() {
  //WaitForDebugger();
  ThreadPool thread_pool(std::thread::hardware_concurrency());
  const auto start1 = std::chrono::high_resolution_clock::now();
  //const auto data = mandelbrot_thread_pool(thread_pool);
  const auto data = mandelbrot_openmp();
  const auto finish1 = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(finish1 - start1) << '\n';
  const auto start = std::chrono::high_resolution_clock::now();
  const auto image = generate_ppm_image(data);
  const auto finish = std::chrono::high_resolution_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start) << '\n';
  std::ofstream(output_image_path, std::ios::binary) << image;
  return 0;
}
