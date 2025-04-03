#include "init.h"

#include <random>

std::pair<std::vector<float>, std::vector<float>> init(size_t size) {
  std::default_random_engine re;
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::vector<float> a(size);
  std::vector<float> b(size);
  for (int i = 0; i < size; ++i) {
    a[i] = dist(re);
    b[i] = dist(re);
  }
  return {a, b};
}
