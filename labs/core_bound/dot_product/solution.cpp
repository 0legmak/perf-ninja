#include "solution.h"

#include <chrono>
#include <numeric>
#include <iostream>

#include <CL/cl_version.h>
#include <CL/opencl.hpp>

float reference_solution(const std::vector<float>& a, const std::vector<float>& b) {
  const auto sz = a.size();
  float res = 0.0;
  for (int i = 0; i < sz; ++i) {
    res += a[i] * b[i];
  }
  return res;
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

const char* dot_product_kernel = R"OpenCL(
  void kernel dot_product(global float* a, global float* b, local float* local_res, int N, global float* final_res) {
    int idx = get_global_id(0);
    int sz = get_global_size(0);
    int lid = get_local_id(0);
    int lsz = get_local_size(0);
    int wg_id = get_group_id(0);
    float res = 0.0f;
    while (idx < N) {
      res += a[idx] * b[idx];
      idx += sz;
    }
    local_res[lid] = res;
    for (int rsz = lsz; rsz > 1; rsz = (rsz + 1) / 2) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (lid < rsz / 2) {
        local_res[lid] += local_res[lid + (rsz + 1) / 2];
      }
    }
    if (lid == 0) {
      final_res[wg_id] = local_res[0];
    }
  }
)OpenCL";

cl::Program program;

} // namespace

void precompile() {
  program = cl::Program(dot_product_kernel, true);
}

float solution(const std::vector<float>& a, const std::vector<float>& b, bool profile) {
  if (profile) {
    cl::CommandQueue::setDefault(cl::CommandQueue(cl::QueueProperties::Profiling));
  }
  const auto N = a.size();
  const auto work_group_size = std::min({
    N,
    256uz,
    cl::Device::getDefault().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()
  });
  const auto work_group_count = std::min(
    (N + work_group_size - 1) / work_group_size,
    4uz * cl::Device::getDefault().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
  );
  cl::Buffer a_buffer(a.begin(), a.end(), true);
  cl::Buffer b_buffer(b.begin(), b.end(), true);
  const auto final_res_bytes = work_group_count * sizeof(float);
  cl::Buffer final_res_buffer(CL_MEM_WRITE_ONLY, final_res_bytes);
  std::chrono::high_resolution_clock::time_point clock_start;
  if (profile) {
    clock_start = std::chrono::high_resolution_clock::now();
  }
  cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, int, cl::Buffer> dot_product(program, "dot_product");
  auto profile_event = dot_product(
    cl::EnqueueArgs(
      cl::NDRange(work_group_count * work_group_size),
      cl::NDRange(work_group_size)
    ),
    a_buffer, b_buffer, cl::Local(work_group_size * sizeof(float)), N, final_res_buffer
  );
  profile_event.wait();
  if (profile) {
    std::cout << "clock time: " << 
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - clock_start) << "\n";
    std::cout << "profile time: " << 
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::nanoseconds(
        profile_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profile_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
      )) << "\n";
  }
  std::vector<float> final_res(work_group_count);
  enqueueReadBuffer(final_res_buffer, CL_TRUE, 0, final_res_bytes, final_res.data());
  return std::accumulate(final_res.begin(), final_res.end(), 0.0f);
}
