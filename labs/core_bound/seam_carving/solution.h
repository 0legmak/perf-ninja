#include "init.h"

#include <vector>
#include <CL/cl_version.h>
#include <CL/opencl.hpp>

std::vector<float> reference_solution(const std::vector<RGB>& input, int width, int height);

class Solution {
public:
  Solution();
  std::vector<float> solution(const std::vector<RGB>& input, int width, int height);
private:
  cl::Context cl_context;
  cl::CommandQueue cl_command_queue;
  cl::Program cl_program;
  std::unique_ptr<cl::KernelFunctor<cl::Buffer, int, cl::Buffer>> calc_energy_kernel;
};
