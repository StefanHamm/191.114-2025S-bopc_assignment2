#include "kernel.hpp"

int checkPointForJuliaSet(int x, int y, Complex c, float scale, int res_x, int res_y, int max_iter, float max_mag, float x_scale, float y_scale) {
    float scaledX = scale * x_scale * (float) (x - res_x / 2) / (res_x / 2);
    float scaledY = scale * y_scale * (float) (y - res_y / 2) / (res_y / 2);

    Complex z(scaledX, scaledY);

    int i = 0;
    for(i = 0; i < max_iter; i++) {
        z = z * z + c;
        if(z.magnitude2() > max_mag)
            break;
    }
    return i;
}

void julia_kernel(float *julia_set, Complex c, float scale, int res_x, int res_y, int max_iter, float max_mag, float x_scale, float y_scale) {
    #pragma omp parallel for collapse(2)
    for(int x = 0; x < res_x; x++) {
        for(int y = 0; y < res_y; y++) {
            // calculate shade for julia set value
            float juliaShade = ((float) checkPointForJuliaSet(x, y, c, scale, res_x, res_y, max_iter, max_mag, x_scale, y_scale)) / max_iter;
            julia_set[x * res_y + y] = juliaShade;
        }
    }
}