// a simple OpenCL kernel which adds two vectors A and B together into a third vector C
kernel void vector_add(global const int* a, global const int* b, global int* c) {
  int id = get_global_id(0);
  c[id] = a[id] + b[id];
}

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

inline float calc_energy(global const unsigned char* pixels, int width, int row, int col) {
  return sqrt(
    squared_gradient(pixels, width, row, col - 1, row, col + 1) +
    squared_gradient(pixels, width, row - 1, col, row + 1, col)
  );
};

kernel void vector_calc_energy(global const unsigned char* pixels, int width, global float* energy) {
  int energy_row = get_global_id(0);
  int energy_col = get_global_id(1);
  int energy_width = width - 2;
  int pixel_row = energy_row + 1;
  int pixel_col = energy_col + 1;
  energy[energy_row * energy_width + energy_col] = calc_energy(pixels, width, pixel_row, pixel_col);
  //energy[energy_row * energy_width + energy_col] = pixels[((pixel_row - 1) * width + pixel_col - 1) * 3];
}
