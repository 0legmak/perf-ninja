inline float get_r(global const unsigned char* pixels, int width, int row, int col) {
  return pixels[(row * width + col) * 3 + 0];
}

inline float get_g(global const unsigned char* pixels, int width, int row, int col) {
  return pixels[(row * width + col) * 3 + 1];
}

inline float get_b(global const unsigned char* pixels, int width, int row, int col) {
  return pixels[(row * width + col) * 3 + 2];
}

inline float sqr(float val) {
  return val * val;
};

inline float squared_gradient(global const unsigned char* pixels, int width, int row1, int col1, int row2, int col2) {
  return
    sqr(get_r(pixels, width, row1, col1) - get_r(pixels, width, row2, col2)) +
    sqr(get_g(pixels, width, row1, col1) - get_g(pixels, width, row2, col2)) +
    sqr(get_b(pixels, width, row1, col1) - get_b(pixels, width, row2, col2));
};

kernel void vector_calc_energy(global const unsigned char* pixels, int width, global float* energy) {
  int row = get_global_id(0) + 1;
  int col = get_global_id(1) + 1;
  energy[row * width + col] = sqrt(
    squared_gradient(pixels, width, row, col - 1, row, col + 1) +
    squared_gradient(pixels, width, row - 1, col, row + 1, col)
  );
}
