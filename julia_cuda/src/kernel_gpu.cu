
#include <iostream>
#include "kernel.hpp"
#include <cuda_runtime.h>

extern int global_block_x, global_block_y;


__global__ void julia_kernel_worker(float *julia_set, Complex c, float scale, int res_x, int res_y, int max_iter, float max_mag, float x_scale, float y_scale) {

    int threadColId = blockIdx.x * blockDim.x + threadIdx.x;
    int threadRowId = blockIdx.y * blockDim.y + threadIdx.y;

    if (threadColId >= res_x || threadRowId >= res_y) return;

    float scaledX = scale * x_scale * (float) (threadColId - res_x / 2) / (res_x / 2);
    float scaledY = scale * y_scale * (float) (threadRowId - res_y / 2) / (res_y / 2);


    Complex z(scaledX, scaledY);

    int i = 0;
    for(i = 0; i < max_iter; i++) {
        z = z * z + c;
        if(z.magnitude2() > max_mag)
            break;
    }
    julia_set[threadRowId*res_x+threadColId] = (float)i/max_iter;
}



void julia_kernel(float *julia_set, Complex c, float scale, int res_x, int res_y, int max_iter, float max_mag, float x_scale, float y_scale) {

    // compute a good default block size
    int device;
    cudaGetDevice(&device)

    cudaDeviceProp prop;   
    cudaGetDeviceProperties( &prop, device);

    printf("Using device %d: %s\n", device, prop.name);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

    dim3 blockShape = dim3(global_block_x, global_block_y);
    dim3 gridShape = dim3( (res_x+blockShape.x-1)/blockShape.x,
                            (res_y+blockShape.y-1)/blockShape.y);

    float *julia_set_d;
    cudaMalloc((void**)&julia_set_d, res_x*res_y*sizeof(float));

    julia_kernel_worker<<<gridShape, blockShape>>>(julia_set_d, c, scale, res_x, res_y, max_iter, max_mag, x_scale, y_scale);
    
    cudaDeviceSynchronize();

    cudaMemcpy(julia_set, julia_set_d, res_x*res_y*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(julia_set_d);
}

