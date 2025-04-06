#include <array>
#include <vector>

constexpr int kWidth = 1000;
constexpr int kHeight = 1000;
constexpr int kRemove = 500;
//constexpr int kWidth = 6;
//constexpr int kHeight = 8 + 6;
//constexpr int kRemove = 3;

using RGB = std::array<unsigned char, 3>;
  
std::vector<RGB> init1();
