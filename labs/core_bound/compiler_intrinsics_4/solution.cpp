#include "solution.h"
#include "const.h"

std::vector<short> mandelbrot() {
  std::vector<short> data(kDataWidth * kDataHeight);
  int data_idx = 0;
  for (int py = 0; py < kDataHeight; ++py) {
    for (int px = 0; px < kDataWidth; ++px) {
      const auto c_x = kMinX + (kMaxX - kMinX) * px / kDataWidth;
      const auto c_y = kMinY + (kMaxY - kMinY) * py / kDataHeight;
      auto z_x = 0.0;
      auto z_y = 0.0;
      int iter = 0;
      for (; iter < kMaxIterations; ++iter) {
        const auto z_xx = z_x * z_x;
        const auto z_yy = z_y * z_y;
        if (z_xx + z_yy > kSquareBound) {
          break;
        }
        z_y = 2 * z_x * z_y + c_y;
        z_x = z_xx - z_yy + c_x;
      }
      data[data_idx++] = iter;
    }
  }
  return data;
}
