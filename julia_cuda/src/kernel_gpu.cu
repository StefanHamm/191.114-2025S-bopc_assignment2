
#include <iostream>
#include "kernel.hpp"

extern int global_block_x, global_block_y;

// Cuda function to compute checko point fo julia set
// this is the worker funcion that will be called for each thread
// the block size etc. will be set int the main julia kernel function this just caluclates the number for the thread
// the blockIdx and blockIdy are the block number in the grid
// the threadIdx is the thread number in the block

__global__ void julia_kernel_worker(float *julia_set, Complex c, float scale, int res_x, int res_y, int max_iter, float max_mag, float x_scale, float y_scale) {

    // compute the thread number in the block
    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    // compute the block number in the grid
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    // compute the global thread number
    int global_thread_x = block_x * blockDim.x + thread_x;
    int global_thread_y = block_y * blockDim.y + thread_y;

    // compute the index in the julia set array
    int index = global_thread_y * res_x + global_thread_x;

    // check if the index is out of bounds
    if (global_thread_x >= res_x || global_thread_y >= res_y) {
        return;
    }

    // compute the complex number for this pixel
    Complex z;
    z.real = (float)global_thread_x / (float)res_x * x_scale - scale;
    z.imag = (float)global_thread_y / (float)res_y * y_scale - scale;

    // iterate to find if it is in the julia set
    for (int i = 0; i < max_iter; i++) {
        if (z.magnitude() > max_mag) {
            julia_set[index] = (float)i / (float)max_iter;
            return;
        }
        z = z * z + c;
    }
    
    // if we get here it is in the julia set
    julia_set[index] = 1.0f;
}



void julia_kernel(float *julia_set, Complex c, float scale, int res_x, int res_y, int max_iter, float max_mag, float x_scale, float y_scale) {

    // compute a good default block size

}

