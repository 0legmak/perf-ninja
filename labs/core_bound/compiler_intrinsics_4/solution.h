#pragma once

#include <string>
#include <vector>

std::vector<short> mandelbrot();
std::string generate_ppm_image(const std::vector<short>& data);
