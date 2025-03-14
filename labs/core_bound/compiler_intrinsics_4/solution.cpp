#include "solution.h"

#include <array>
#include <numeric>
#include <stdexcept>
#include <print>

#include <complex>
#include <sstream>
#include <ostream>

namespace {

constexpr auto kMaxIterations = 2000;
constexpr auto kSquareBound = 4.0;
constexpr auto kImageWidth = 1280;
constexpr auto kImageHeight = 720;
constexpr auto kDataWidth = kImageWidth + 2;
constexpr auto kDataHeight = kImageHeight + 2;
constexpr auto kCenterX = -0.743643135;
constexpr auto kCenterY = 0.131825963;
constexpr auto kDiameterX = 0.000014628;
constexpr auto kDiameterY = kDiameterX / kImageWidth * kImageHeight;
constexpr auto kMinX = kCenterX - kDiameterX / 2;
constexpr auto kMaxX = kCenterX + kDiameterX / 2;
constexpr auto kMinY = kCenterY - kDiameterY / 2;
constexpr auto kMaxY = kCenterY + kDiameterY / 2;
constexpr auto kScale = 1.0;
constexpr auto kOffset = 0.0;
constexpr double kGaussKernel[3][3] = {
  { 1.0 / 16, 2.0 / 16, 1.0 / 16 },
  { 2.0 / 16, 4.0 / 16, 2.0 / 16 },
  { 1.0 / 16, 2.0 / 16, 1.0 / 16 }
};

struct Point {
  double x;
  double y;
};

constexpr size_t kColorCnt = 3;

using RGB = std::array<uint8_t, kColorCnt>;
using Gradient = std::array<std::vector<Point>, kColorCnt>;

constexpr auto kBackgroundColor = RGB{0, 0, 0};
const Gradient kGradient = {{
  { {0.0, 0.0}, {0.7, 1.0}, {1.0, 0.0} },
  { {0.0, 0.0}, {0.5, 1.0}, {1.0, 0.0} },
  { {0.0, 0.0}, {0.3, 1.0}, {1.0, 0.0} },
}};

class CubicBezier {
public:
  CubicBezier() = default;
  CubicBezier(const Point p0, const Point p1, const Point p2, const Point p3) noexcept
    : p0(p0), p1(p1), p2(p2), p3(p3) 
  {}
  Point calculate(double t) const noexcept {
    const double t_2 = t * t;
    const double t_3 = t_2 * t;
    const double u = 1 - t;
    const double u_2 = u * u;
    const double u_3 = u_2 * u;
    Point result;
    result.x = u_3 * p0.x + 3 * u_2 * t * p1.x + 3 * u * t_2 * p2.x + t_3 * p3.x;
    result.y = u_3 * p0.y + 3 * u_2 * t * p1.y + 3 * u * t_2 * p2.y + t_3 * p3.y;
    return result;
  }
private:
  Point p0;
  Point p1;
  Point p2;
  Point p3;
};

double lerp(const Point p1, const Point p2, double x) {
  const auto dx = p2.x - p1.x;
  const auto dy = p2.y - p1.y;
  return p1.y + dy * (x - p1.x) / dx;
}

std::vector<RGB> calc_color_map() {
  constexpr int size = kMaxIterations;
  std::array<std::vector<Point>, kColorCnt> interpolated;
  for (int c = 0; c < kColorCnt; ++c) {
    interpolated[c].resize(size);
    int c_idx = 0;
    double c_x1, c_x2;
    CubicBezier bezier;
    auto update_curve = [&]() {
      const auto p1 = kGradient[c][c_idx];
      const auto p2 = kGradient[c][c_idx + 1];
      const double mid_x = std::midpoint(p1.x, p2.x);
      bezier = CubicBezier(p1, {mid_x, p1.y}, {mid_x, p2.y}, p2);
      c_x1 = p1.x;
      c_x2 = p2.x;
    };
    update_curve();
    for (int idx = 0; idx < size; ++idx) {
      const double x = 1.0 * idx / (size - 1);
      if (x > kGradient[c][c_idx + 1].x) {
        do {
          ++c_idx;
        } while (x > kGradient[c][c_idx + 1].x);
        update_curve();
      }
      const double t = (x - c_x1) / (c_x2 - c_x1);
      interpolated[c][idx] = bezier.calculate(t);
    }
  }
  std::vector<RGB> colors(size);
  for (int c = 0; c < kColorCnt; ++c) {
    int in_idx = 0;
    for (int out_idx = 0; out_idx < size; ++out_idx) {
      const double x = 1.0 * out_idx / (size - 1);
      while (in_idx < size && interpolated[c][in_idx].x < x) {
        ++in_idx;
      }
      double res;
      if (in_idx + 1 < size) {
        res = lerp(interpolated[c][in_idx], interpolated[c][in_idx + 1], x);
      } else {
        res = interpolated[c].back().y;
      }
      colors[out_idx][c] = std::min(static_cast<int>(res * 256), 255);
    }
  }
  std::vector<RGB> result(size + 1);
  for (int i = 0; i < size; ++i) {
    result[i] = colors[static_cast<int>(i * kScale + kOffset * size) % size];
  }
  result[size] = kBackgroundColor;
  return result;
}

RGB gaussian_blur(const RGB input[3][3]) {
  std::array<double, 3> output{};
  for (int x = 0; x < 3; ++x) {
    for (int y = 0; y < 3; ++y) {
      for (int c = 0; c < 3; ++c) {
        output[c] += input[x][y][c] * kGaussKernel[x][y];
      }
    }
  }
  return {
    static_cast<uint8_t>(output[0]),
    static_cast<uint8_t>(output[1]),
    static_cast<uint8_t>(output[2]),
  };
}

}  // namespace

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

std::string generate_ppm_image(const std::vector<short>& data) {
  std::string out;
  out.append("P3\n");
  out.append(std::to_string(kImageWidth));
  out.push_back(' ');
  out.append(std::to_string(kImageHeight));
  out.push_back('\n');
  out.append("255\n");
  RGB color_matrix[3][3];
  const auto color_map = calc_color_map();
  for (int py = 0; py < kImageHeight; ++py) {
    for (int x = 0; x < 2; ++x) {
      for (int y = 0; y < 3; ++y) {
        color_matrix[x + 1][y] = color_map[data[(py + y) * kDataWidth + x]];
      }
    }
    for (int px = 0; px < kImageWidth; ++px) {
      for (int y = 0; y < 3; ++y) {
        color_matrix[0][y] = color_matrix[1][y];
        color_matrix[1][y] = color_matrix[2][y];
        color_matrix[2][y] = color_map[data[(py + y) * kDataWidth + px + 2]];
      }
      const auto color = gaussian_blur(color_matrix);
      out.append(std::to_string(color[0]));
      out.push_back(' ');
      out.append(std::to_string(color[1]));
      out.push_back(' ');
      out.append(std::to_string(color[2]));
      out.push_back(" \n"[px == kImageWidth - 1]);
    }
  }
  return out;
}
