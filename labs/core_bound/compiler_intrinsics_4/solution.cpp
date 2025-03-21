#include "solution.h"
#include "const.h"
#include "thread_pool.h"

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>

#include <array>
#include <bit>
#include <cmath>

namespace {
#if defined(__AVX512F__)
using Vec = __m512d;
using VecInt = __m512i;
constexpr auto& vec_setzero = _mm512_setzero_pd;
constexpr auto& vec_set1 = _mm512_set1_pd;
constexpr auto& vec_set1_int = _mm512_set1_epi64;
constexpr auto& vec_load = _mm512_loadu_pd;
constexpr auto& vec_store = _mm512_storeu_pd;
constexpr auto& vec_store_int = _mm512_storeu_si512;
constexpr auto& vec_add = _mm512_add_pd;
constexpr auto& vec_add_int = _mm512_add_epi64;
constexpr auto& vec_sub = _mm512_sub_pd;
constexpr auto& vec_mul = _mm512_mul_pd;
constexpr auto vec_blend = [](auto a, auto b, auto m) { return _mm512_mask_blend_pd(m, a, b); };
constexpr auto vec_cmpeq_int64 = [](auto a, auto b) { return _mm512_cmpeq_epi64_mask(a, b); };
constexpr auto vec_cmpgt_double = [](auto a, auto b) { return _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ); };
constexpr auto vec_or_mask = [](auto a, auto b) { return a | b; };
constexpr auto vec_movemask = [](auto m) { return m; };
#elif defined(__AVX2__)
using Vec = __m256d;
using VecInt = __m256i;
constexpr auto& vec_setzero = _mm256_setzero_pd;
constexpr auto& vec_set1 = _mm256_set1_pd;
constexpr auto& vec_set1_int = _mm256_set1_epi64x;
constexpr auto& vec_load = _mm256_loadu_pd;
constexpr auto& vec_store = _mm256_storeu_pd;
constexpr auto& vec_store_int = _mm256_storeu_si256;
constexpr auto& vec_add = _mm256_add_pd;
constexpr auto& vec_add_int = _mm256_add_epi64;
constexpr auto& vec_sub = _mm256_sub_pd;
constexpr auto& vec_mul = _mm256_mul_pd;
constexpr auto vec_blend = [](auto a, auto b, auto m) { return _mm256_blendv_pd(a, b, m); };
constexpr auto vec_cmpeq_int64 = [](auto a, auto b) { return _mm256_cmpeq_epi64(a, b); };
constexpr auto vec_cmpgt_double = [](auto a, auto b) { return _mm256_cmp_pd(a, b, _CMP_GT_OQ); };
constexpr auto vec_or_mask = [](auto a, auto b) { return _mm256_or_si256(a, b); };
constexpr auto vec_movemask = [](auto m) { return _mm256_movemask_pd(m); };
#else
using Vec = __m128d;
using VecInt = __m128i;
constexpr auto& vec_setzero = _mm_setzero_pd;
constexpr auto& vec_set1 = _mm_set1_pd;
constexpr auto& vec_set1_int = _mm_set1_epi64x;
constexpr auto& vec_load = _mm_loadu_pd;
constexpr auto& vec_store = _mm_storeu_pd;
constexpr auto& vec_store_int = _mm_storeu_si128;
constexpr auto& vec_add = _mm_add_pd;
constexpr auto& vec_add_int = _mm_add_epi64;
constexpr auto& vec_sub = _mm_sub_pd;
constexpr auto& vec_mul = _mm_mul_pd;
constexpr auto vec_blend = [](auto a, auto b, auto m) { return _mm_blendv_pd(a, b, m); };
constexpr auto vec_cmpeq_int64 = [](auto a, auto b) { return _mm_cmpeq_epi64(a, b); };
constexpr auto vec_cmpgt_double = [](auto a, auto b) { return _mm_cmpgt_pd(a, b); };
constexpr auto vec_or_mask = [](auto a, auto b) { return _mm_or_si128(a, b); };
constexpr auto vec_movemask = [](auto m) { return _mm_movemask_pd(m); };
#endif
constexpr auto kVecSize = sizeof(Vec) / sizeof(double);
constexpr auto kChunkSize = 1000;
}  // namespace

std::vector<short> mandelbrot(ImplType impl_type) {
  constexpr size_t data_size = kDataWidth * kDataHeight;
  std::vector<short> data(data_size);
  const auto squared_bound = vec_set1(kSquareBound);
  const auto max_iter = vec_set1_int(kMaxIterations);
  const auto iter_inc = vec_set1_int(1);
  
  auto original = [&]() {
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
  };

  auto process_chunk = [&](size_t begin, size_t end) {
    auto [py, px] = std::div(begin, kDataWidth);
    std::array<double, kVecSize> c_x_arr;
    std::array<double, kVecSize> c_y_arr;
    std::array<size_t, kVecSize> res_idx;
    size_t data_idx = begin;
    size_t res_used = 0;
    auto next_data_item = [&](int idx) {
      if (data_idx < end) {
        c_x_arr[idx] = std::lerp(kMinX, kMaxX, 1.0 * px / kDataWidth);
        c_y_arr[idx] = std::lerp(kMinY, kMaxY, 1.0 * py / kDataHeight);
        if (++px == kDataWidth) {
          px = 0;
          ++py;
        }
        res_idx[idx] = data_idx;
        ++data_idx;
        ++res_used;
      } else {
        c_x_arr[idx] = 0.0;
        c_y_arr[idx] = 0.0;
        res_idx[idx] = -1;
      }
    };
    for (int i = 0; i < kVecSize; ++i) {
      next_data_item(i);
    }
    auto c_x = vec_load(c_x_arr.data());
    auto c_y = vec_load(c_y_arr.data());
    auto z_x = vec_setzero();
    auto z_y = vec_setzero();
    auto iter_cnt = vec_setzero();
    while (true) {
      const auto max_iter_mask = vec_cmpeq_int64(iter_cnt, max_iter);
      auto z_xx = vec_mul(z_x, z_x);
      auto z_yy = vec_mul(z_y, z_y);
      const auto squared_bound_mask = vec_cmpgt_double(vec_add(z_xx, z_yy), squared_bound);
      const auto cond_mask = vec_or_mask(max_iter_mask, squared_bound_mask);
      if (uint8_t mask = vec_movemask(cond_mask); mask) {
        std::array<uint64_t, kVecSize> iter_cnt_arr;
        vec_store_int((VecInt*)iter_cnt_arr.data(), iter_cnt);
        for (; mask; mask &= mask - 1) {
          const auto ridx = std::countr_zero(mask);
          if (res_idx[ridx] != -1) {
            data[res_idx[ridx]] = iter_cnt_arr[ridx];
            if (--res_used == 0) {
              return;
            }
          }
          next_data_item(ridx);
        }
        z_x = vec_blend(z_x, vec_setzero(), cond_mask);
        z_y = vec_blend(z_y, vec_setzero(), cond_mask);
        z_xx = vec_blend(z_xx, vec_setzero(), cond_mask);
        z_yy = vec_blend(z_yy, vec_setzero(), cond_mask);
        c_x = vec_blend(c_x, vec_load(c_x_arr.data()), cond_mask);
        c_y = vec_blend(c_y, vec_load(c_y_arr.data()), cond_mask);
        iter_cnt = vec_blend(iter_cnt, vec_setzero(), cond_mask);
      }
      const auto z_xy = vec_mul(z_x, z_y);
      z_x = vec_add(vec_sub(z_xx, z_yy), c_x);
      z_y = vec_add(vec_add(z_xy, z_xy), c_y);
      iter_cnt = vec_add_int(iter_cnt, iter_inc);
    }
  };
  switch (impl_type) {
    case ImplType::kOriginal:
      original();
      break;
    case ImplType::kVectorized:
      process_chunk(0, data_size);
      break;
    case ImplType::kThreadPool: {
      ThreadPool& thread_pool = get_thread_pool();
      std::vector<std::future<void>> threaded_results;
      for (size_t idx = 0; idx < data_size; idx += kChunkSize) {
        threaded_results.push_back(thread_pool.enqueue(process_chunk, idx, std::min(data_size, idx + kChunkSize)));
      }
      for (auto& res : threaded_results) {
        res.wait();
      }
      break;
    }
    case ImplType::kOpenMP: {
      size_t curr_chunk_id = 0;
      #pragma omp parallel
      {
        while (true) {
          size_t chunk_id;
          #pragma omp atomic capture relaxed
          {
            chunk_id = curr_chunk_id;
            curr_chunk_id++;
          }
          if (chunk_id * kChunkSize >= data_size) {
            break;
          }
          process_chunk(chunk_id * kChunkSize, std::min(data_size, (chunk_id + 1) * kChunkSize));
        }
      }
      break;
    }
  }
  return data;
}
