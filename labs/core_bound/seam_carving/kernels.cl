#define kTileSize 8
#define kPixelSize 3
#define kBorderEnergy 1000.0f
#define kMaxDist FLT_MAX

inline float get_r(local const unsigned char* tile, int row, int col) {
  return tile[(row * (kTileSize + 2) + col) * kPixelSize + 0];
}

inline float get_g(local const unsigned char* tile, int row, int col) {
  return tile[(row * (kTileSize + 2) + col) * kPixelSize + 1];
}

inline float get_b(local const unsigned char* tile, int row, int col) {
  return tile[(row * (kTileSize + 2) + col) * kPixelSize + 2];
}

inline float sqr(float val) {
  return val * val;
};

inline float squared_gradient(local const unsigned char* tile, int row1, int col1, int row2, int col2) {
  return
    sqr(get_r(tile, row1, col1) - get_r(tile, row2, col2)) +
    sqr(get_g(tile, row1, col1) - get_g(tile, row2, col2)) +
    sqr(get_b(tile, row1, col1) - get_b(tile, row2, col2));
};

kernel void calc_energy(global const unsigned char* pixels, int buffer_width, int width, int height, global float* energy) {
  int row_tile_idx = get_global_id(0) / (kTileSize + 2);
  int col_tile_idx = get_global_id(1) / (kTileSize + 2);
  int tile_row = get_local_id(0);
  int tile_col = get_local_id(1);
  int row = row_tile_idx * kTileSize + tile_row;
  int col = col_tile_idx * kTileSize + tile_col;
  local unsigned char tile[(kTileSize + 2) * (kTileSize + 2) * kPixelSize];
  for (int i = 0; i < kPixelSize; ++i) {
    tile[(tile_row * (kTileSize + 2) + tile_col) * kPixelSize + i] =
      pixels[(row * buffer_width + col) * kPixelSize + i];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (row == 0 || row >= height - 1 || col == 0 || col >= width - 1) {
    energy[row * buffer_width + col] = kBorderEnergy;
  } else if (tile_row > 0 && tile_row <= kTileSize && tile_col > 0 && tile_col <= kTileSize) {
    energy[row * buffer_width + col] = sqrt(
      squared_gradient(tile, tile_row, tile_col - 1, tile_row, tile_col + 1) +
      squared_gradient(tile, tile_row - 1, tile_col, tile_row + 1, tile_col)
    );
  }
}

kernel void calc_dist(global const float* energy, int buffer_width, int width, int height, global float* dist, global char* prev) {
  const int col = get_global_id(0);
  dist[col] = 0.0f;
  prev[col] = 0;
  dist[(height - 1) * width + col] = 0.0f;
  prev[(height - 1) * width + col] = 0;
  for (int row = 1; row < height - 1; ++row) {
    barrier(CLK_LOCAL_MEM_FENCE);
    global float* d = &dist[row * width + col];
    global char* p = &prev[row * width + col];
    *p = 0;
    if (col == 0 || col == width - 1) {
      *d = kMaxDist;
    } else {
      global const float* d_l = &dist[(row - 1) * width + col - 1];
      global const float* d_r = &dist[(row - 1) * width + col + 1];
      *d = dist[(row - 1) * width + col];
      if (*d > *d_l) {
          *d = *d_l;
          *p = -1;
      }
      if (*d > *d_r) {
          *d = *d_r;
          *p = 1;
      }
      *d += energy[row * buffer_width + col];
    }
  }
}

kernel void find_seam(global const float* dist, global const char* prev, int width, int height, global int* seam) {
  const int last_row = height - 2;
  int min_col = 1;
  float min_dist = dist[last_row * width + min_col];
  for (int col = 1; col < width - 1; ++col) {
    if (min_dist > dist[last_row * width + col]) {
      min_col = col;
      min_dist = dist[last_row * width + min_col];
    }
  }
  for (int row = last_row, col = min_col; row > 0; --row) {
    seam[row] = col;
    col = col + prev[row * width + col];
  }
  seam[0] = seam[1];
  seam[height - 1] = seam[last_row];
}

inline float get_color(global const unsigned char* pixels, int buffer_width, int row, int col, int color_idx) {
  return pixels[(row * buffer_width + col) * kPixelSize + color_idx];
}

inline float squared_gradient2(global const unsigned char* pixels, int buffer_width, int row1, int col1, int row2, int col2) {
  return
    sqr(get_color(pixels, buffer_width, row1, col1, 0) - get_color(pixels, buffer_width, row2, col2, 0)) +
    sqr(get_color(pixels, buffer_width, row1, col1, 1) - get_color(pixels, buffer_width, row2, col2, 1)) +
    sqr(get_color(pixels, buffer_width, row1, col1, 2) - get_color(pixels, buffer_width, row2, col2, 2));
};

kernel void delete_seam(global const int* seam, int buffer_width, int width, global unsigned char* pixels, global float* energy) {
  const int row = get_global_id(0) + 1;
  for (int col = seam[row]; col < width - 1; ++col) {
    for (int i = 0; i < kPixelSize; ++i) {
      pixels[(row * buffer_width + col) * kPixelSize + i] = pixels[(row * buffer_width + col + 1) * kPixelSize + i];
    }
  }
  for (int col = seam[row]; col < width - 1; ++col) {
    energy[row * buffer_width + col] = energy[row * buffer_width + col + 1];
  }
  --width;
  for (int col = seam[row] - 1; col <= seam[row]; ++col) {
    if (col == 0 || col == width - 1) {
      energy[row * buffer_width + col] = kBorderEnergy;
    } else {
      energy[row * buffer_width + col] = sqrt(
        squared_gradient2(pixels, buffer_width, row, col - 1, row, col + 1) +
        squared_gradient2(pixels, buffer_width, row - 1, col, row + 1, col)
      );
    }
  }
}
