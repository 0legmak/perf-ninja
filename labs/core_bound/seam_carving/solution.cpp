#include "solution.h"
#include "data_paths.h"

#include <fstream>
#include <mdspan>
#include <sstream>

#include <CL/cl_version.h>
#include <CL/opencl.hpp>

namespace {
  constexpr float kBorderEnergy = 1000.0f;
}

class Reference : public ISolution {
public:
  std::vector<RGB> process(const std::vector<RGB>& input, int width, int height, int remove_cnt) override {
    const std::array<size_t, 2> buffer_strides{(size_t)width, 1};
    auto create_buffer_span = [height, &width, buffer_strides](auto* ptr) {
      std::dextents<size_t, 2> extents(height, width);
      const auto mapping = std::layout_stride::mapping(extents, buffer_strides);
      return std::mdspan(ptr, mapping);
    };
    // calculate dual-gradient energy function for pixels
    std::vector<RGB> pixels_buffer(input);
    auto pixels = create_buffer_span(pixels_buffer.data());
    auto calc_energy = [&pixels](int row, int col) {
      auto squared_gradient = [](const RGB c1, const RGB c2) {
        auto squared_diff = [](float v1, float v2) {
          return (v1 - v2) * (v1 - v2);
        };
        return squared_diff(c1[0], c2[0]) + squared_diff(c1[1], c2[1]) + squared_diff(c1[2], c2[2]);
      };
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
    std::vector<float> dist_buffer(width * height);
    std::vector<char> prev_buffer(width * height);
    std::vector<int> seam(height);
    while (remove_cnt--) {
      auto dist = create_buffer_span(dist_buffer.data());
      auto prev = create_buffer_span(prev_buffer.data());
      // solve shortest path in DAG
      for (int row = 1; row < height - 1; ++row) {
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
      // find a vertical seam (minimal energy path from top to bottom)
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
      // delete vertical seam
      for (int row = 1; row < height - 1; ++row) {
        for (int col = seam[row]; col < width - 1; ++col) {
          pixels[row, col] = pixels[row, col + 1];
          energy[row, col] = energy[row, col + 1];
        }
      }
      --width;
      pixels = create_buffer_span(pixels_buffer.data());
      energy = create_buffer_span(energy_buffer.data());
      // recalculate energy
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
};

std::unique_ptr<ISolution> reference_solution() {
  return std::make_unique<Reference>();
}

namespace {
  std::string load_file(const std::string& file_path) {
    std::ifstream fileStream(file_path);
    std::stringstream buffer;
    buffer << fileStream.rdbuf();
    return buffer.str();
  }

  template <typename T>
  class BufferMapping {
  public:
    BufferMapping(const cl::Buffer& buffer, cl_map_flags flags) : buffer(buffer) {
      const auto buffer_size = buffer.getInfo<CL_MEM_SIZE>();
      cl::Event event;
      buffer_ptr = static_cast<T*>(enqueueMapBuffer(buffer, CL_FALSE, flags, 0, buffer_size, nullptr, &event));
      event.wait();
    }
    ~BufferMapping() {
      try {
        unmap();
      } catch (...) {
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
      enqueueUnmapMemObject(buffer, buffer_ptr, nullptr, &event);
      event.wait();
      buffer_ptr = nullptr;
    }
  private:
    const cl::Buffer& buffer;
    T* buffer_ptr;
  };
} // namespace

class Solution : public ISolution {
public:
  Solution() :
    program(load_file(kKernelSourcePath), true),
    calc_energy_kernel(program, "calc_energy"),
    calc_dist_kernel(program, "calc_dist"),
    find_seam_kernel(program, "find_seam"),
    delete_seam_kernel(program, "delete_seam")
  {
  }
  std::vector<RGB> process(const std::vector<RGB>& input, int width, int height, int remove_cnt) override {
    constexpr auto kTileSize = 8;
    const auto tile_col_cnt = (width - 2 + kTileSize - 1) / kTileSize;
    const auto tile_row_cnt = (height - 2 + kTileSize - 1) / kTileSize;
    const auto buffer_width = tile_col_cnt * kTileSize + 2;
    const auto buffer_height = tile_row_cnt * kTileSize + 2;
    cl::Buffer pixel_buffer(CL_MEM_READ_WRITE, buffer_width * buffer_height * sizeof(RGB));
    BufferMapping<RGB> pixel_buffer_mapping(pixel_buffer, CL_MAP_WRITE);
    for (int i = 0; i < height; ++i) {
      std::copy(input.begin() + i * width, input.begin() + (i + 1) * width, pixel_buffer_mapping.ptr() + i * buffer_width);
    }
    pixel_buffer_mapping.unmap();
    cl::Buffer energy_buffer(CL_MEM_READ_WRITE, buffer_width * buffer_height * sizeof(float));
    cl::Buffer dist_buffer(CL_MEM_READ_WRITE, width * height * sizeof(float));
    cl::Buffer prev_buffer(CL_MEM_READ_WRITE, width * height * sizeof(char));
    cl::Buffer seam_buffer(CL_MEM_READ_WRITE, height * sizeof(int));
    calc_energy_kernel(
      cl::EnqueueArgs(
        cl::NDRange(tile_row_cnt * (kTileSize + 2), tile_col_cnt * (kTileSize + 2)),
        cl::NDRange(kTileSize + 2, kTileSize + 2)
      ),
      pixel_buffer,
      buffer_width,
      width,
      height,
      energy_buffer
    );
    while (remove_cnt--) {
      calc_dist_kernel(
        cl::EnqueueArgs(cl::NDRange(width), cl::NDRange(width)),
        energy_buffer,
        buffer_width,
        width,
        height,
        dist_buffer,
        prev_buffer
      );
      find_seam_kernel(
        cl::EnqueueArgs(cl::NDRange(1), cl::NDRange(1)),
        dist_buffer,
        prev_buffer,
        width,
        height,
        seam_buffer
      );
      delete_seam_kernel(
        cl::EnqueueArgs(cl::NDRange(height - 2), cl::NDRange(height - 2)),
        seam_buffer,
        buffer_width,
        width,
        pixel_buffer,
        energy_buffer
      );
      --width;
    }
    std::vector<RGB> result(width * height);
    BufferMapping<RGB> result_buffer_mapping(pixel_buffer, CL_MAP_READ);
    for (int r = 0; r < height; ++r) {
      std::copy(result_buffer_mapping.ptr() + r * buffer_width, result_buffer_mapping.ptr() + r * buffer_width + width, result.begin() + r * width);
    }
    result_buffer_mapping.unmap();
    return result;
  }
private:
  cl::Program program;
  cl::KernelFunctor<cl::Buffer, int, int, int, cl::Buffer> calc_energy_kernel;
  cl::KernelFunctor<cl::Buffer, int, int, int, cl::Buffer, cl::Buffer> calc_dist_kernel;
  cl::KernelFunctor<cl::Buffer, cl::Buffer, int, int, cl::Buffer> find_seam_kernel;
  cl::KernelFunctor<cl::Buffer, int, int, cl::Buffer, cl::Buffer> delete_seam_kernel;
};

std::unique_ptr<ISolution> solution() {
  return std::make_unique<Solution>();
}
