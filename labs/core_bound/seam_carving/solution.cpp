#include "solution.h"
#include "data_paths.h"

#include <chrono>
#include <fstream>
#include <numeric>
#include <mdspan>
#include <print>
#include <sstream>
#include <string>
#include <span>

namespace {
  constexpr float kBorderEnergy = 1000.0f;

  std::string load_file(const std::string& file_path) {
    std::ifstream fileStream(file_path);
    std::stringstream buffer;
    buffer << fileStream.rdbuf();
    return buffer.str();
  }
}

std::vector<RGB> reference_solution(const std::vector<RGB>& input, int width, int height, int remove_cnt) {
  const auto start = std::chrono::high_resolution_clock::now();

  const std::array<size_t, 2> buffer_strides{(size_t)width, 1};
  auto create_buffer_span = [height, &width, buffer_strides](auto* ptr) {
    std::dextents<size_t, 2> extents(height, width);
    const auto mapping = std::layout_stride::mapping(extents, buffer_strides);
    return std::mdspan(ptr, mapping);
  };

  std::vector<RGB> pixels_buffer(input);
  auto pixels = create_buffer_span(pixels_buffer.data());
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
  std::vector<float> energy_buffer(width * height);
  auto energy = create_buffer_span(energy_buffer.data());
  for (int row = 1; row < height - 1; ++row) {
    energy[row, 0] = energy[row, width - 1] = kBorderEnergy;
    for (int col = 1; col < width - 1; ++col) {
      energy[row, col] = calc_energy(row, col);
    }
  }
  for (int col = 0; col < width; ++col) {
    energy[0, col] = energy[height - 1, col] = kBorderEnergy;
  }

  std::println("ref time: {}", 
    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count()
  );
  //for (int r = 0; r < height; ++r) {
  //  for (int c = 0; c < width; ++c) {
  //    std::print("{:8.5} ", energy[r, c]);
  //  }
  //  std::println("");
  //}

  std::vector<float> dist_buffer(width * height);
  std::vector<char> prev_buffer(width * height);
  while (remove_cnt--) {
    auto dist = create_buffer_span(dist_buffer.data());
    auto prev = create_buffer_span(prev_buffer.data());
    for (int row = 1; row < height; ++row) {
      dist[row, 0] = dist[row, width - 1] = std::numeric_limits<float>::max();
      for (int col = 1; col < width - 1; ++col) {
        dist[row, col] = dist[row - 1, col];
        if (dist[row, col] > dist[row - 1, col - 1]) {
            dist[row, col] = dist[row - 1, col - 1];
            prev[row, col] = -1;
        }
        if (dist[row, col] > dist[row - 1, col + 1]) {
            dist[row, col] = dist[row - 1, col + 1];
            prev[row, col] = 1;
        }
        dist[row, col] += energy[row, col];
      }
    }

    std::vector<int> seam(height);
    const auto last_row = height - 2;
    auto min_col = 1;
    auto min_dist = dist[last_row, min_col];
    for (int col = 1; col < width - 1; ++col) {
      if (min_dist > dist[last_row, col]) {
        min_col = col;
        min_dist = dist[last_row, min_col];
      }
    }
    for (int row = last_row, col = min_col; row > 0; --row) {
      seam[row] = col;
      col = col + prev[row, col];
    }
    seam[0] = seam[1];
    seam[height - 1] = seam[last_row];

    for (int row = 1; row < height - 1; ++row) {
      for (int col = seam[row]; col < width - 1; ++col) {
        pixels[row, col] = pixels[row, col + 1];
        energy[row, col] = energy[row, col + 1];
      }
    }
    --width;
    pixels = create_buffer_span(pixels_buffer.data());
    energy = create_buffer_span(energy_buffer.data());

    for (int row = 1; row < height - 1; ++row) {
      for (const int col : { seam[row] - 1, seam[row] }) {
        if (col == 0 || col == width - 1) {
            energy[row, col] = kBorderEnergy;
        } else {
            energy[row, col] = calc_energy(row, col);
        }
      }
    }
  }

  std::vector<RGB> result_buffer(width * height);
  std::mdspan<RGB, std::dextents<size_t, 2>> result(result_buffer.data(), height, width);
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      result[row, col] = pixels[row, col];
    }
  }
  return result_buffer;
}

Solution::Solution() {
  cl_context = cl::Context::getDefault();
  //for (const auto& device : cl_context.getInfo<CL_CONTEXT_DEVICES>()) {
  //  std::println("Using device: {}", device.getInfo<CL_DEVICE_NAME>());
  //}
  cl_command_queue = cl::CommandQueue(cl_context, CL_QUEUE_PROFILING_ENABLE);
  const auto kernel_sources = load_file(kernels_cl_path);
  cl_program = cl::Program(cl_context, kernel_sources, /* build */ true);
  calc_energy_kernel = std::make_unique<cl::KernelFunctor<cl::Buffer, int, cl::Buffer>>(cl_program, "vector_calc_energy");
}

template <typename T>
class BufferMapping {
public:
  BufferMapping(
    const cl::Buffer& buffer,
    const cl::CommandQueue& command_queue,
    cl::size_type size,
    cl_map_flags flags
  ) : buffer(buffer), command_queue(command_queue) 
  {
    cl::Event event;
    buffer_ptr = static_cast<T*>(command_queue.enqueueMapBuffer(buffer, CL_FALSE, flags, 0, size, nullptr, &event));
    event.wait();
    std::println("buffer map profile time: {}", 
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::nanoseconds(
        event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
      )).count()
    );
  }

  ~BufferMapping() {
    try {
      unmap();
    } catch (...) {
      return;
    }
  }
 
  T* ptr() {
    return buffer_ptr;
  }

  void unmap() {
    if (buffer_ptr == nullptr) {
      return;
    }
    cl::Event event;
    command_queue.enqueueUnmapMemObject(buffer, buffer_ptr, nullptr, &event);
    event.wait();
    std::println("buffer unmap profile time: {}", 
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::nanoseconds(
        event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
      )).count()
    );
    buffer_ptr = nullptr;
  }

private:
  const cl::Buffer& buffer;
  const cl::CommandQueue& command_queue;
  T* buffer_ptr;
};

class Timer {
public:
  Timer() {
    pt = std::chrono::high_resolution_clock::now();
  }
  void mark(const std::string& label) {
    const auto ct = std::chrono::high_resolution_clock::now();
    std::println("{} time: {}", label, std::chrono::duration_cast<std::chrono::milliseconds>(ct - pt).count());
    pt = ct;
  }
private:
  std::chrono::high_resolution_clock::time_point pt;
};

std::vector<float> Solution::solution(const std::vector<RGB>& input, int width, int height) {
  Timer timer;
  constexpr auto kWorkGroupSz = 8;
  const auto buffer_width = (width - 2 + kWorkGroupSz - 1) / kWorkGroupSz * kWorkGroupSz + 2;
  const auto buffer_height = (height - 2 + kWorkGroupSz - 1) / kWorkGroupSz * kWorkGroupSz + 2;
  const auto pixel_buffer_size = buffer_width * buffer_height * sizeof(RGB);
  cl::Buffer pixel_buffer(cl_context, CL_MEM_READ_ONLY, pixel_buffer_size);
  timer.mark("pixel_buffer");
  BufferMapping<RGB> pixel_buffer_mapping(pixel_buffer, cl_command_queue, pixel_buffer_size, CL_MAP_WRITE);
  timer.mark("pixel_buffer_mapping");
  for (int r = 0; r < height; ++r) {
    std::copy(input.begin() + r * width, input.begin() + (r + 1) * width, pixel_buffer_mapping.ptr() + r * buffer_width);
  }
  timer.mark("pixel_buffer_copy");
  pixel_buffer_mapping.unmap();
  timer.mark("pixel_buffer_unmapping");

  const auto energy_buffer_bytes = buffer_width * buffer_height * sizeof(float);
  cl::Buffer energy_buffer(cl_context, CL_MEM_WRITE_ONLY, energy_buffer_bytes);
  timer.mark("energy_buffer");

  auto calc_energy_event = (*calc_energy_kernel)(
    cl::EnqueueArgs(
      cl_command_queue,
      cl::NDRange(buffer_height - 2, buffer_width - 2),
      cl::NDRange(kWorkGroupSz, kWorkGroupSz)
    ),
    pixel_buffer,
    buffer_width,
    energy_buffer
  );
  calc_energy_event.wait();
  std::println("vector_calc_energy exec time: {}", 
    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::nanoseconds(
      calc_energy_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - calc_energy_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
    )).count()
  );
  timer.mark("calc_energy_kernel");

  std::vector<float> energy(width * height);
  BufferMapping<float> energy_buffer_mapping(energy_buffer, cl_command_queue, energy_buffer_bytes, CL_MAP_READ);
  timer.mark("energy_buffer_mapping");
  for (int r = 1; r < height - 1; ++r) {
    std::copy(energy_buffer_mapping.ptr() + r * buffer_width, energy_buffer_mapping.ptr() + r * buffer_width + width, energy.begin() + r * width);
  }
  timer.mark("energy_buffer_copy");
  energy_buffer_mapping.unmap();
  timer.mark("energy_buffer_unmapping");
  for (int row = 0; row < height; ++row) {
    energy[row * width + 0] = energy[row * width + width - 1] = kBorderEnergy;
  }
  for (int col = 0; col < width; ++col) {
    energy[0 * width + col] = energy[(height - 1) * width + col] = kBorderEnergy;
  }

  //for (int r = 0; r < height; ++r) {
  //  for (int c = 0; c < width; ++c) {
  //    std::print("{:8.5} ", energy[r * width + c]);
  //  }
  //  std::println("");
  //}

  timer.mark("end");
  return energy;
}
