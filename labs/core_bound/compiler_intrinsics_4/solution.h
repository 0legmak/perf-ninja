#include <vector>

class ThreadPool;

enum ImplType {
  kOriginal,
  kVectorized,
  kThreadPool,
  kOpenMP
};

std::vector<short> mandelbrot(int image_width, int image_height, ImplType impl_type);
