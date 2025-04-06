#include "init.h"

#include <memory>
#include <vector>

class ISolution {
public:
  virtual ~ISolution() {};
  virtual std::vector<RGB> process(const std::vector<RGB>& input, int width, int height, int remove_cnt) = 0;
};

std::unique_ptr<ISolution> reference_solution();
std::unique_ptr<ISolution> solution();
