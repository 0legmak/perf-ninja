#include <vector>

class ThreadPool;

std::vector<short> mandelbrot();
std::vector<short> mandelbrot_thread_pool(ThreadPool& thread_pool);
std::vector<short> mandelbrot_openmp();
