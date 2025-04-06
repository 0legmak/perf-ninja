#include "init.h"

#include <random>

std::vector<RGB> init1() {
  std::default_random_engine re(0);
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<RGB> data(kWidth * kHeight);
  for (int r = 0; r < kHeight; ++r) {
    for (int c = 0; c < kWidth; ++c) {
      for (int i = 0; i < 3; ++i) {
        data[r * kWidth + c][i] = dist(re);
      }
    }
  }
  return data;
}
