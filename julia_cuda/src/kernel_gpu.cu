
#include <iostream>
#include "kernel.hpp"

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
    //printf("threadColId: %d threadRowId: %d scaledX: %f scaledY: %f\n", threadColId, threadRowId, scaledX, scaledY);
    julia_set[threadRowId*res_x+threadColId] = i/max_iter;
    // return i;

}



void julia_kernel(float *julia_set, Complex c, float scale, int res_x, int res_y, int max_iter, float max_mag, float x_scale, float y_scale) {

    // compute a good default block size

    dim3 blockShape = dim3(4, 8);
    dim3 gridShape = dim3( (res_x+blockShape.x-1)/blockShape.x,
                            (res_y+blockShape.y-1)/blockShape.y);

    //cudaMallocManaged((void**)&julia_set, res_x*res_y*sizeof(float));
    float *julia_set_d;
    cudaMalloc((void**)&julia_set_d, res_x*res_y*sizeof(float));

    julia_kernel_worker<<<gridShape, blockShape>>>(julia_set_d, c, scale, res_x, res_y, max_iter, max_mag, x_scale, y_scale);
    cudaError_t err = cudaGetLastError();
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    
    cudaDeviceSynchronize();

    //for (int i = 0; i < res_x*res_y; i++) {
    //    printf("julia_set[%d]: %f\n", i, julia_set_d[i]);
    //}
    cudaMemcpy(julia_set, julia_set_d, res_x*res_y*sizeof(float), cudaMemcpyDeviceToHost);
    
    //for (int i = 0; i < res_x*res_y; i++) {
    //    printf("julia_set[%d]: %f\n", i, julia_set_d[i]);
    //}
    for (int i = 0; i < res_x*res_y; i++) {
        printf("julia_set[%d]: %f\n", i, julia_set[i]);
    }
    
    err = cudaGetLastError();
    printf("CUDA memcpy error: %s\n", cudaGetErrorString(err));
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
    }

    //cudaDeviceSynchronize();

    cudaFree(julia_set_d);
}

