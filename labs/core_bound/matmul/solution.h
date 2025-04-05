#include <memory>
#include <vector>

class ISolution {
public:
  virtual ~ISolution() {};
  virtual void set_input(const std::vector<float>& a, const std::vector<float>& b, int N, int K, int M) = 0;
  virtual void run_kernel() = 0;
  virtual std::vector<float> get_output() = 0;
};

std::unique_ptr<ISolution> reference_solution();
std::unique_ptr<ISolution> solution();
