#include "solution.h"
#include "const.h"
#include "thread_pool.h"

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

#include <array>
#include <bit>

namespace {
#if defined(__AVX512F__)
using Vec = __m512d;
constexpr auto& vec_setzero = _mm512_setzero_pd;
constexpr auto& vec_set1 = _mm512_set1_pd;
constexpr auto& vec_load = _mm512_loadu_pd;
constexpr auto& vec_store = _mm512_storeu_pd;
constexpr auto& vec_add = _mm512_add_pd;
constexpr auto& vec_sub = _mm512_sub_pd;
constexpr auto& vec_mul = _mm512_mul_pd;
constexpr auto vec_cmpgt_mask = [](auto a, auto b) { return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ); };
#elif defined(__AVX2__)
using Vec = __m256d;
constexpr auto& vec_setzero = _mm256_setzero_pd;
constexpr auto& vec_set1 = _mm256_set1_pd;
constexpr auto& vec_load = _mm256_loadu_pd;
constexpr auto& vec_store = _mm256_storeu_pd;
constexpr auto& vec_add = _mm256_add_pd;
constexpr auto& vec_sub = _mm256_sub_pd;
constexpr auto& vec_mul = _mm256_mul_pd;
constexpr auto vec_cmpgt_mask = [](auto a, auto b) { return _mm256_movemask_pd(_mm256_cmp_pd(a, b, _CMP_GT_OQ)); };
#else
using Vec = __m128d;
constexpr auto& vec_setzero = _mm_setzero_pd;
constexpr auto& vec_set1 = _mm_set1_pd;
constexpr auto& vec_load = _mm_loadu_pd;
constexpr auto& vec_store = _mm_storeu_pd;
constexpr auto& vec_add = _mm_add_pd;
constexpr auto& vec_sub = _mm_sub_pd;
constexpr auto& vec_mul = _mm_mul_pd;
constexpr auto vec_cmpgt_mask = [](auto a, auto b) { return _mm_movemask_pd(_mm_cmpgt_pd(a, b)); };
#endif
constexpr auto kVecSize = sizeof(Vec) / sizeof(double);
}  // namespace

#define SOLUTION
#ifdef SOLUTION

std::vector<short> mandelbrot(ThreadPool& thread_pool) {
  constexpr size_t data_size = (kDataWidth * kDataHeight + kVecSize - 1) / kVecSize * kVecSize;
  std::vector<short> data(data_size);
  auto job = [&data](size_t begin, size_t end) -> void {
    auto px = begin % kDataWidth;
    auto py = begin / kDataWidth;
    const auto squared_bound = vec_set1(kSquareBound);
    for (int data_idx = begin; data_idx < end; data_idx += kVecSize) {
      std::array<double, kVecSize> c_x_src;
      std::array<double, kVecSize> c_y_src;
      for (int i = 0; i < kVecSize; ++i) {
        c_x_src[i] = std::lerp(kMinX, kMaxX, 1.0 * px / kDataWidth);
        c_y_src[i] = std::lerp(kMinY, kMaxY, 1.0 * py / kDataHeight);
        if (++px == kDataWidth) {
          px = 0;
          ++py;
        }
      }
      const auto c_x = vec_load(c_x_src.data());
      const auto c_y = vec_load(c_y_src.data());
      auto z_x = vec_setzero();
      auto z_y = vec_setzero();
      std::array<int, kVecSize> res;
      res.fill(kMaxIterations);
      auto res_cnt = 0;
      for (int iter_cnt = 0; iter_cnt < kMaxIterations; ++iter_cnt) {
        const auto z_xx = vec_mul(z_x, z_x);
        const auto z_yy = vec_mul(z_y, z_y);
        for (unsigned mask = vec_cmpgt_mask(vec_add(z_xx, z_yy), squared_bound); mask; ) {
          const auto res_idx = std::countr_zero(mask);
          res_cnt += res[res_idx] == kMaxIterations;
          res[res_idx] = std::min(res[res_idx], iter_cnt);
          mask -= 1 << res_idx;
        }
        if (res_cnt == kVecSize) {
          break;
        }
        const auto z_xy = vec_mul(z_x, z_y);
        z_x = vec_add(vec_sub(z_xx, z_yy), c_x);
        z_y = vec_add(vec_add(z_xy, z_xy), c_y);
      }
      std::copy(res.begin(), res.end(), data.begin() + data_idx);
    }
  };
  constexpr size_t threaded_job_size = (1000 + kVecSize - 1) / kVecSize * kVecSize;
  std::vector<std::future<void>> threaded_results;
  for (size_t idx = 0; idx < data_size; idx += threaded_job_size) {
    threaded_results.push_back(thread_pool.enqueue(job, idx, std::min(data_size, idx + threaded_job_size)));
  }
  for (auto& res : threaded_results) {
    res.wait();
  }
  return data;
}

#elif defined(SOLUTION2)

std::vector<short> mandelbrot() {
  constexpr size_t data_size = (kDataWidth * kDataHeight + kVecSize - 1) / kVecSize * kVecSize;
  std::vector<short> data(data_size);
  auto px = 0;
  auto x = kMinX;
  auto y = kMinY;
  const auto dx = (kMaxX - kMinX) / (kDataWidth - 1);
  const auto dy = (kMaxY - kMinY) / (kDataHeight - 1);
  const auto squared_bound = vec_set1(kSquareBound);
  const auto init_res = vec_set1(kMaxIterations);
  const auto one = vec_set1(1.0);
  for (int data_idx = 0; data_idx < data_size; data_idx += kVecSize) {
    std::array<double, kVecSize> c_x_src;
    std::array<double, kVecSize> c_y_src;
    for (int i = 0; i < kVecSize; ++i) {
      c_x_src[i] = x;
      c_y_src[i] = y;
      if (++px == kDataWidth) {
        px = 0;
        x = kMinX;
        y += dy;
      } else {
        x += dx;
      }
    }
    const auto c_x = vec_load(c_x_src.data());
    const auto c_y = vec_load(c_y_src.data());
    auto z_x = vec_setzero();
    auto z_y = vec_setzero();
    auto res = init_res;
    auto iter_vec = vec_setzero();
    
#if defined(__AVX512F__)
    auto res_mask = (1u << kVecSize) - 1;
#else
    auto res_mask = _mm_cmpeq_pd(vec_setzero(), vec_setzero());
#endif

    for (int iter_cnt = 0; iter_cnt < kMaxIterations; ++iter_cnt) {
      const auto z_xx = vec_mul(z_x, z_x);
      const auto z_yy = vec_mul(z_y, z_y);
      const auto z_xx_yy = vec_add(z_xx, z_yy);

#if defined(__AVX512F__)
      if (res_mask == 0) {
        break;
      }
      const auto mask = _mm512_mask_cmp_pd_mask(res_mask, z_xx_yy, squared_bound, _CMP_GT_OQ);
      res = _mm512_mask_mov_pd(res, mask, iter_vec);
      res_mask &= ~mask;
#else
      if (_mm_movemask_pd(res_mask) == 0) {
        break;
      }
      const auto bound_mask = _mm_cmpgt_pd(z_xx_yy, squared_bound);
      res = _mm_blendv_pd(res, iter_vec, _mm_and_pd(bound_mask, res_mask));
      res_mask = _mm_andnot_pd(bound_mask, res_mask);
#endif

      iter_vec = vec_add(iter_vec, one);
      const auto z_xy = vec_mul(z_x, z_y);
      z_x = vec_add(vec_sub(z_xx, z_yy), c_x);
      z_y = vec_add(vec_add(z_xy, z_xy), c_y);
    }
    std::array<double, kVecSize> out;
    vec_store(out.data(), res);
    for (int i = 0; i < kVecSize; ++i) {
      data[data_idx + i] = out[i];
    }
  }
  return data;
}

#else

std::vector<short> mandelbrot() {
  std::vector<short> data(kDataWidth * kDataHeight);
  auto data_idx = 0;
  for (int py = 0; py < kDataHeight; ++py) {
    for (int px = 0; px < kDataWidth; ++px) {
      const auto c_x = kMinX + (kMaxX - kMinX) * px / kDataWidth;
      const auto c_y = kMinY + (kMaxY - kMinY) * py / kDataHeight;
      auto z_x = 0.0;
      auto z_y = 0.0;
      int iter_cnt = 0;
      for (; iter_cnt < kMaxIterations; ++iter_cnt) {
        const auto z_xx = z_x * z_x;
        const auto z_yy = z_y * z_y;
        if (z_xx + z_yy > kSquareBound) {
          break;
        }
        const auto z_xy = z_x * z_y;
        z_x = z_xx - z_yy + c_x;
        z_y = z_xy + z_xy + c_y;
      }
      data[data_idx++] = iter_cnt;
    }
  }
  return data;
}

#endif
