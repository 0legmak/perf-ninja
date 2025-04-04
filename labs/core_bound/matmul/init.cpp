#include "init.h"

#include <random>

std::pair<std::vector<float>, std::vector<float>> init(int N, int K, int M) {
  std::default_random_engine re;
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::vector<float> a(N * K);
  std::vector<float> b(K * M);
  for (auto& v : a) {
    v = dist(re);
  }
  for (auto& v : b) {
    v = dist(re);
  }
  return {a, b};
}
