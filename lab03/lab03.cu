#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

#include "dummy_helper.hpp"
#include "image.hpp"

#define MAX_N_CLASSES 32
#define BLOCK_SIZE_REDUCE 256
#define GRID_SIZE_REDUCE 16

__constant__ float avg_d[MAX_N_CLASSES * 3];
__constant__ uint32_t cov_matrices_norms_d[MAX_N_CLASSES];
__constant__ float reverse_cov_matrices_d[MAX_N_CLASSES * 9];


float avg_h[MAX_N_CLASSES * 3];
uint32_t cov_matrices_norms_h[MAX_N_CLASSES];
float reverse_cov_matrices_h[MAX_N_CLASSES * 9];


__device__
void sum_vectors(float *a, float *b, float *result, size_t dim1, size_t dim2)
{
    for (size_t i = 0; i < dim1; ++i)
        for (size_t j = 0; j < dim2; ++j)
        {
            size_t idx = i * dim2 + j;
            result[idx] = a[idx] + b[idx];
        }
}


__device__
void sum_vectors_v(volatile float *a, volatile float *b, volatile float *result, size_t dim1, size_t dim2)
{
    for (size_t i = 0; i < dim1; ++i)
        for (size_t j = 0; j < dim2; ++j)
        {
            size_t idx = i * dim2 + j;
            result[idx] = a[idx] + b[idx];
        }
}


__device__
void mul_vectors(
        float *a, 
        float *b, 
        float *result, 
        size_t a_dim1, 
        size_t a_dim2,
        size_t b_dim1,
        size_t b_dim2
)
{
    for (size_t i = 0; i < b_dim2; ++i)
        for (size_t j = 0; j < a_dim1; ++j)
        {
            float current_value = 0;

            for (size_t k = 0; k < a_dim2; ++k)
                current_value += a[j * a_dim2 + k] * b[k * b_dim2 + i];

            result[j * b_dim2 + i] = current_value;
        }
}


__global__ 
void reduce_avg(
        uchar4 *image, 
        size_t width, 
        uint32_t *samples,
        size_t start,
        float *output) 
{
    __shared__ float sdata[BLOCK_SIZE_REDUCE][3];

    size_t tid = threadIdx.x;

    sdata[tid][0] = 0;
    sdata[tid][1] = 0;
    sdata[tid][2] = 0;

    size_t i = start + 1 + blockIdx.x * BLOCK_SIZE_REDUCE * 2 + 2 * tid;
    size_t offset = BLOCK_SIZE_REDUCE * 2 * GRID_SIZE_REDUCE * 2;
    
    size_t n = start + 1 + 2 * samples[start];
    while (i < n) 
    { 
        if (i + 1 + BLOCK_SIZE_REDUCE < n)
        {
            float a[3], b[3];
            size_t position      = samples[i + 1] * width + samples[i],
                   next_position = samples[i + 1 + BLOCK_SIZE_REDUCE] * width + samples[i + BLOCK_SIZE_REDUCE];

            a[0] = image[position].x;
            a[1] = image[position].y;
            a[2] = image[position].z;

            b[0] = image[next_position].x;
            b[1] = image[next_position].y;
            b[2] = image[next_position].z;
            
            sum_vectors(a, b, sdata[tid], 1, 3);
        }
        else
        {
            float a[3];
            size_t position = samples[i + 1] * width + samples[i];

            a[0] = image[position].x;
            a[1] = image[position].y;
            a[2] = image[position].z;
            
            sum_vectors_v(a, sdata[tid], sdata[tid], 1, 3);
        }

        i += offset;
    }
    
    __syncthreads();
    
    if (BLOCK_SIZE_REDUCE >= 512) { if (tid < 256) { sum_vectors_v(sdata[tid], sdata[tid + 256], sdata[tid], 1, 3); } __syncthreads(); }
    if (BLOCK_SIZE_REDUCE >= 256) { if (tid < 128) { sum_vectors_v(sdata[tid], sdata[tid + 128], sdata[tid], 1, 3); } __syncthreads(); }
    if (BLOCK_SIZE_REDUCE >= 128) { if (tid < 64)  { sum_vectors_v(sdata[tid], sdata[tid +  64], sdata[tid], 1, 3); } __syncthreads(); }
    
    if (tid < 32)
    {
        if (BLOCK_SIZE_REDUCE >= 64) { sum_vectors_v(sdata[tid], sdata[tid + 32], sdata[tid], 1, 3); __syncthreads();}
        if (BLOCK_SIZE_REDUCE >= 32) { sum_vectors_v(sdata[tid], sdata[tid + 16], sdata[tid], 1, 3); __syncthreads();}
        if (BLOCK_SIZE_REDUCE >= 16) { sum_vectors_v(sdata[tid], sdata[tid +  8], sdata[tid], 1, 3); __syncthreads();}
        if (BLOCK_SIZE_REDUCE >= 8)  { sum_vectors_v(sdata[tid], sdata[tid +  4], sdata[tid], 1, 3); __syncthreads();}
        if (BLOCK_SIZE_REDUCE >= 4)  { sum_vectors_v(sdata[tid], sdata[tid +  2], sdata[tid], 1, 3); __syncthreads();}
        if (BLOCK_SIZE_REDUCE >= 2)  { sum_vectors_v(sdata[tid], sdata[tid +  1], sdata[tid], 1, 3); __syncthreads();}

        /* if (BLOCK_SIZE_REDUCE >= 64) { sum_vectors_v(sdata[tid], sdata[tid + 32], sdata[tid], 1, 3); } */
        /* if (BLOCK_SIZE_REDUCE >= 32) { sum_vectors_v(sdata[tid], sdata[tid + 16], sdata[tid], 1, 3); } */
        /* if (BLOCK_SIZE_REDUCE >= 16) { sum_vectors_v(sdata[tid], sdata[tid +  8], sdata[tid], 1, 3); } */
        /* if (BLOCK_SIZE_REDUCE >= 8)  { sum_vectors_v(sdata[tid], sdata[tid +  4], sdata[tid], 1, 3); } */
        /* if (BLOCK_SIZE_REDUCE >= 4)  { sum_vectors_v(sdata[tid], sdata[tid +  2], sdata[tid], 1, 3); } */
        /* if (BLOCK_SIZE_REDUCE >= 2)  { sum_vectors_v(sdata[tid], sdata[tid +  1], sdata[tid], 1, 3); } */

    }

    if (tid == 0)
    {
        output[blockIdx.x * 3 + 0] = sdata[0][0];
        output[blockIdx.x * 3 + 1] = sdata[0][1];
        output[blockIdx.x * 3 + 2] = sdata[0][2];
    }
}


__global__
void calc_avg_cov_reduced(
        uchar4 *image,
        size_t width,
        uint32_t *samples,
        float *reduced_buffer,
        size_t n_classes)
{
    
}

int submain()
{
    std::string input_name,
                output_name;

    std::cin >> input_name >> output_name;

    size_t n_classes;
    std::cin >> n_classes;
    
    std::vector<uint32_t> samples_h;

    for (size_t i = 0; i < n_classes; ++i)
    {
        uint32_t n_observations;
        std::cin >> n_observations;
        
        samples_h.push_back(n_observations);

        for (size_t i = 0; i < n_observations; ++i)
        {
            uint32_t column, row;
            std::cin >> column >> row;

            samples_h.push_back(column);
            samples_h.push_back(row);
        }
    }
    
    CudaMemory<uint32_t> samples_d(samples_h.size());

    samples_d.memcpy(
        samples_h.data(),
        cudaMemcpyHostToDevice
    );

    Image<uchar4> input_image_h(input_name);

    CudaMemory<uchar4> input_image_d(input_image_h.count());

    input_image_d.memcpy(
        input_image_h.buffer.data(),
        cudaMemcpyHostToDevice
    );

    CudaMemory<float> reduced_buffer_d(GRID_SIZE_REDUCE * 9);

    size_t start = 0;

    for (size_t c = 0; c < n_classes; ++c)
    {
        cudaError_t err = cudaSuccess;

        reduce_avg<<<GRID_SIZE_REDUCE, BLOCK_SIZE_REDUCE>>>(
            input_image_d.get(), 
            input_image_h.width, 
            samples_d.get(),
            start, 
            reduced_buffer_d.get()
        );

        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            std::cerr << "ERROR: Failed to launch vector_max kernel (error "
                      << cudaGetErrorString(err)
                      << ")!"
                      << std::endl;
            exit(EXIT_FAILURE);
        }
        
        std::vector<float> reduced_buffer_h(GRID_SIZE_REDUCE * 9);
        
        reduced_buffer_d.memcpy(
            reduced_buffer_h.data(), 
            cudaMemcpyDeviceToHost
        );

        float r = 0,
              g = 0,
              b = 0;

        size_t n_observations = samples_h[start];

        for (size_t i = 0; i < GRID_SIZE_REDUCE; ++i)
        {
            r += reduced_buffer_h[i * 3 + 0] / n_observations;
            g += reduced_buffer_h[i * 3 + 1] / n_observations;
            b += reduced_buffer_h[i * 3 + 2] / n_observations;
        }

        avg_h[c * 3 + 0] = r;
        avg_h[c * 3 + 1] = g;
        avg_h[c * 3 + 2] = b;

        std::cout << r 
                  << ' '
                  << g
                  << ' '
                  << b
                  << std::endl;

        start += samples_h[start]*2 + 1;
    }
    
    return 0;
}

int main()
{
    try
    {
        submain();
    }
    catch (const std::exception &err)
    {
        std::cout << "ERROR:" << err.what() << std::endl;
    }

    return 0;
}
