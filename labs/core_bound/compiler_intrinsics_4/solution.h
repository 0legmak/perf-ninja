#include <vector>

class ThreadPool;

enum ImplType {
  kOriginal,
  kVectorized,
  kThreadPool,
  kOpenMP
};

std::vector<short> mandelbrot(ImplType impl_type);
