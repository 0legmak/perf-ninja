#include "solution.h"

#include <CL/cl_version.h>
#include <CL/opencl.hpp>

namespace {

const char* matmul_source = R"OpenCL(
  void kernel matmul(global float* a, global float* b, int N, int K, int M, global float* res) {
    for (int row = get_global_id(0); row < N; row += get_global_size(0)) {
      for (int col = get_global_id(1); col < M; col += get_global_size(1)) {
        float r = 0.0f;
        for (int k = 0; k < K; ++k) {
          r += a[row * K + k] * b[k * M + col];
        }
        res[row * M + col] = r;
      }
    }
  }
)OpenCL";

} // namespace

class Reference : public ISolution {
public:
  Reference() : program(matmul_source, true), matmul(program, "matmul") {
  }
  void set_input(const std::vector<float>& a, const std::vector<float>& b, int N, int K, int M) override {
    a_buffer = cl::Buffer(a.begin(), a.end(), true);
    b_buffer = cl::Buffer(b.begin(), b.end(), true);
    result_bytes = N * M * sizeof(float);
    result_buffer = cl::Buffer(CL_MEM_WRITE_ONLY, result_bytes);
    this->N = N;
    this->K = K;
    this->M = M;
  }
  void run_kernel() override {
    matmul(
      cl::EnqueueArgs(cl::NDRange(N, M)),
      a_buffer, b_buffer, N, K, M, result_buffer
    ).wait();
  }
  std::vector<float> get_output() override {
    std::vector<float> result(N * M);
    enqueueReadBuffer(result_buffer, CL_TRUE, 0, result_bytes, result.data());
    return result;
  }
private:
  cl::Program program;
  cl::KernelFunctor<cl::Buffer, cl::Buffer, int, int, int, cl::Buffer> matmul;
  size_t result_bytes = 0;
  cl::Buffer a_buffer;
  cl::Buffer b_buffer;
  int N = 0;
  int K = 0;
  int M = 0;
  cl::Buffer result_buffer;
};

std::unique_ptr<ISolution> reference_solution() {
  return std::make_unique<Reference>();
}

namespace {

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

const char* matmul_tiled_source = R"OpenCL(
  void kernel matmul(
    global float* a,
    global float* b,
    int N,
    int K,
    int M,
    int tile_size,
    local float* a_tile,
    local float* b_tile,
    global float* res
  ) {
    int tile_row = get_local_id(0);
    int tile_col = get_local_id(1);
    for (int row = get_global_id(0); row < N; row += get_global_size(0)) {
      for (int col = get_global_id(1); col < M; col += get_global_size(1)) {
        float r = 0.0f;
        for (int tile_offset = 0; tile_offset < K; tile_offset += tile_size) {
          a_tile[tile_row * tile_size + tile_col] = a[row * K + tile_offset + tile_col];
          b_tile[tile_row * tile_size + tile_col] = b[(tile_offset + tile_row) * M + col];
          barrier(CLK_LOCAL_MEM_FENCE);
          for (int k = 0; k < tile_size; ++k) {
            r += a_tile[tile_row * tile_size + k] * b_tile[k * tile_size + tile_col];
          }
          barrier(CLK_LOCAL_MEM_FENCE);
        }
        res[row * M + col] = r;
      }
    }
  }
)OpenCL";

} // namespace

class Solution : public ISolution {
public:
  Solution() : program(matmul_tiled_source, true), matmul(program, "matmul") {
  }
  void set_input(const std::vector<float>& a, const std::vector<float>& b, int N, int K, int M) override {
    this->N = N;
    this->K = K;
    this->M = M;
    auto next_multiple = [](auto a, auto b) {
      return (a + b - 1) / b * b;
    };
    N_buf = next_multiple(N, kTileSize);
    K_buf = next_multiple(K, kTileSize);
    M_buf = next_multiple(M, kTileSize);
    a_buffer = cl::Buffer(CL_MEM_READ_ONLY, N_buf * K_buf * sizeof(float));
    BufferMapping<float> a_mapping(a_buffer, CL_MAP_WRITE);
    for (int i = 0; i < N; ++i) {
      std::copy(a.begin() + i * K, a.begin() + (i + 1) * K, a_mapping.ptr() + i * K_buf);
      std::fill(a_mapping.ptr() + i * K_buf + K, a_mapping.ptr() + (i + 1) * K_buf, 0.0f);
    }
    std::fill(a_mapping.ptr() + N * K_buf, a_mapping.ptr() + N_buf * K_buf, 0.0f);
    a_mapping.unmap();
    b_buffer = cl::Buffer(CL_MEM_READ_ONLY, K_buf * M_buf * sizeof(float));
    BufferMapping<float> b_mapping(b_buffer, CL_MAP_WRITE);
    for (int i = 0; i < K; ++i) {
      std::copy(b.begin() + i * M, b.begin() + (i + 1) * M, b_mapping.ptr() + i * M_buf);
      std::fill(b_mapping.ptr() + i * M_buf + M, b_mapping.ptr() + (i + 1) * M_buf, 0.0f);
    }
    std::fill(b_mapping.ptr() + K * M_buf, b_mapping.ptr() + K_buf * M_buf, 0.0f);
    b_mapping.unmap();
    const auto result_bytes = N_buf * M_buf * sizeof(float);
    result_buffer = cl::Buffer(CL_MEM_WRITE_ONLY, result_bytes);
  }
  void run_kernel() override {
    const auto local_buffer = cl::Local(kTileSize * kTileSize * sizeof(float));
    matmul(
      cl::EnqueueArgs(cl::NDRange(N_buf, M_buf), cl::NDRange(kTileSize, kTileSize)),
      a_buffer, b_buffer, N_buf, K_buf, M_buf, kTileSize, local_buffer, local_buffer, result_buffer
    ).wait();
  }
  std::vector<float> get_output() override {
    std::vector<float> result(N * M);
    BufferMapping<float> result_mapping(result_buffer, CL_MAP_READ);
    for (int i = 0; i < N; ++i) {
      std::copy(result_mapping.ptr() + i * M_buf, result_mapping.ptr() + i * M_buf + M, result.begin() + i * M);
    }
    result_mapping.unmap();
    return result;
  }
private:
  static constexpr int kTileSize = 8;
  cl::Program program;
  cl::KernelFunctor<
    cl::Buffer,
    cl::Buffer,
    int,
    int,
    int,
    int,
    cl::LocalSpaceArg,
    cl::LocalSpaceArg,
    cl::Buffer
  > matmul;
  cl::Buffer a_buffer;
  cl::Buffer b_buffer;
  int N = 0;
  int K = 0;
  int M = 0;
  int N_buf = 0;
  int K_buf = 0;
  int M_buf = 0;
  cl::Buffer result_buffer;
};

std::unique_ptr<ISolution> solution() {
  return std::make_unique<Solution>();
}
