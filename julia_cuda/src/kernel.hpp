
#ifndef JULIA_KERNEL
#define JULIA_KERNEL

#if MODE == 1
#include "complex_gpu.cuh"
#else
#include "complex.hpp"
#endif

void julia_kernel(float *julia_set, Complex c, float scale, int res_x, int res_y, int max_iter, float max_mag, float x_scale, float y_scale);

#endif
