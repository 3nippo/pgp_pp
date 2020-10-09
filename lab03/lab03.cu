#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

#include "dummy_helper.hpp"
#include "image.hpp"

#define MAX_N_CLASSES 32
#define REDUCTION_BLOCK_SIZE 256
#define REDUCTION_GRID_SIZE 16
#define DIM1 3
#define DIM2 1


__constant__ float avg[MAX_N_CLASSES][3];
__constant__ float cov_matrices_norms[MAX_N_CLASSES];
__constant__ float inversed_cov_matrices[MAX_N_CLASSES][9];


__host__ __device__
void sum_vectors(
        float *a, 
        float *b, 
        float *result, 
        size_t dim1,
        size_t dim2
)
{
    for (size_t i = 0; i < dim1; ++i)
        for (size_t j = 0; j < dim2; ++j)
        {
            size_t idx = i * dim2 + j;
            result[idx] = a[idx] + b[idx];
        }
}


__device__
void sum_vectors_v(
        volatile float *a, 
        volatile float *b, 
        volatile float *result, 
        size_t dim1,
        size_t dim2
)
{
    for (size_t i = 0; i < dim1; ++i)
        for (size_t j = 0; j < dim2; ++j)
        {
            size_t idx = i * dim2 + j;
            result[idx] = a[idx] + b[idx];
        }
}


__host__ __device__
void multiply_vectors(
        float *a, 
        float *b, 
        float *result, 
        size_t a_dim1,
        size_t a_dim2,
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


float matrix_norm(float *m, size_t dim1, size_t dim2) { return 0;  }


void inverse_matrix(float *m, float *result, size_t dim) {}


class PixelReader
{
protected:
    template <typename PixelType>
    __host__ __device__
    void read_pixel(float *v, PixelType pixel);
};


template <>
__host__ __device__
void PixelReader::read_pixel(float *v, uchar4 pixel)
{
    v[0] = pixel.x;
    v[1] = pixel.y;
    v[2] = pixel.z;
}


template<
    size_t dim1,
    size_t dim2
>
class InitAvg : public PixelReader
{
public:
    __host__ __device__
    void operator()(
            uchar4 *image, 
            size_t width, 
            uint32_t *samples,
            size_t current_class,
            float *sdata,
            size_t pixel_sample_position,
            size_t sample_end
    )
    {
        if (pixel_sample_position + 1 + REDUCTION_BLOCK_SIZE < sample_end)
        {
            float a[dim1*dim2], b[dim1*dim2];
            size_t position      = samples[pixel_sample_position + 1] * width + samples[pixel_sample_position],
                   next_position = samples[pixel_sample_position + 1 + REDUCTION_BLOCK_SIZE] * width + samples[pixel_sample_position + REDUCTION_BLOCK_SIZE];
            
            read_pixel(a, image[position]);
            read_pixel(b, image[next_position]);
            
            sum_vectors(a, b, sdata, dim1, dim2);
        }
        else
        {
            float a[dim1*dim2];
            size_t position = samples[pixel_sample_position + 1] * width + samples[pixel_sample_position];

            read_pixel(a, image[position]);
            
            sum_vectors(a, sdata, sdata, dim1, dim2);
        }
    }
};


template <
    size_t dim1,
    size_t dim2
>
class InitCov : PixelReader
{
public:
    __host__ __device__
    void operator()(
            uchar4 *image, 
            size_t width, 
            uint32_t *samples,
            size_t current_class,
            float *sdata,
            size_t pixel_sample_position,
            size_t sample_end
    )
    {
        if (pixel_sample_position + 1 + REDUCTION_BLOCK_SIZE < sample_end)
        {
            float a[dim1*dim2], b[dim1*dim2];
            size_t position      = samples[pixel_sample_position + 1] * width + samples[pixel_sample_position],
                   next_position = samples[pixel_sample_position + 1 + REDUCTION_BLOCK_SIZE] * width + samples[pixel_sample_position + REDUCTION_BLOCK_SIZE];

            read_pixel(a, image[position]);
            read_pixel(b, image[next_position]);

            float a_difference_with_avg[dim1*dim2], b_difference_with_avg[dim1*dim2];

            sum_vectors(a, avg[current_class], a_difference_with_avg, dim1, dim2);
            sum_vectors(b, avg[current_class], b_difference_with_avg, dim1, dim2);

            float ma[dim1*dim1], mb[dim1*dim1];

            multiply_vectors(
                a_difference_with_avg, 
                a_difference_with_avg,
                ma,
                dim1,
                dim2,
                dim1
            );

            multiply_vectors(
                b_difference_with_avg, 
                b_difference_with_avg,
                mb,
                dim1,
                dim2,
                dim1
            );
            
            sum_vectors(ma, mb, sdata, dim1, dim1);
        }
        else
        {
            float a[dim1*dim2];
            size_t position = samples[pixel_sample_position + 1] * width + samples[pixel_sample_position];

            read_pixel(a, image[position]);
            
            float difference_with_avg[dim1*dim2];

            sum_vectors(a, avg[current_class], difference_with_avg, dim1, dim2);

            float m[dim1*dim1];

            multiply_vectors(
                difference_with_avg, 
                difference_with_avg,
                m,
                dim1,
                dim2,
                dim1
            );

            sum_vectors(m, sdata, sdata, dim1, dim1);
        }
    }
};


template<
    size_t dim1,
    size_t dim2
>
class Deinit
{
public:
    __host__ __device__
    void operator()(float *sdata, float *reduction_buffer)
    {
        const size_t offset = dim1 * dim2;
        
        for (size_t i = 0; i < offset; ++i)
            reduction_buffer[blockIdx.x * offset + i] = sdata[i];
    }
};


template <
    size_t dim1,
    size_t dim2,
    typename FInit,
    typename FDeinit,
    typename PixelPointerType
>
__device__ 
void reduce(
        PixelPointerType image, 
        size_t width, 
        uint32_t *samples,
        size_t current_class,
        size_t start,
        float *reduction_buffer,
        FInit init,
        FDeinit deinit
) 
{
    __shared__ float sdata[REDUCTION_BLOCK_SIZE][dim1*dim2];

    size_t tid = threadIdx.x;
    
    for (size_t i = 0; i < dim1 * dim2; ++i)
        sdata[tid][i] = 0;

    size_t pixel_sample_position = start + 1 + blockIdx.x * REDUCTION_BLOCK_SIZE * 2 + 2 * tid;
    size_t offset = REDUCTION_BLOCK_SIZE * 2 * REDUCTION_GRID_SIZE * 2;
    
    size_t sample_end = start + 1 + 2 * samples[start];
    while (pixel_sample_position < sample_end) 
    { 
        init(
            image,
            width,
            samples,
            current_class,
            sdata[tid],
            pixel_sample_position,
            sample_end
        );          

        pixel_sample_position += offset;
    }
    
    __syncthreads();
    
    if (REDUCTION_BLOCK_SIZE >= 512) { if (tid < 256) { sum_vectors(sdata[tid], sdata[tid + 256], sdata[tid], dim1, dim2); } __syncthreads(); }
    if (REDUCTION_BLOCK_SIZE >= 256) { if (tid < 128) { sum_vectors(sdata[tid], sdata[tid + 128], sdata[tid], dim1, dim2); } __syncthreads(); }
    if (REDUCTION_BLOCK_SIZE >= 128) { if (tid < 64)  { sum_vectors(sdata[tid], sdata[tid +  64], sdata[tid], dim1, dim2); } __syncthreads(); }
    
    if (tid < 32)
    {
        if (REDUCTION_BLOCK_SIZE >= 64) { sum_vectors_v(sdata[tid], sdata[tid + 32], sdata[tid], dim1, dim2); }
        if (REDUCTION_BLOCK_SIZE >= 32) { sum_vectors_v(sdata[tid], sdata[tid + 16], sdata[tid], dim1, dim2); }
        if (REDUCTION_BLOCK_SIZE >= 16) { sum_vectors_v(sdata[tid], sdata[tid +  8], sdata[tid], dim1, dim2); }
        if (REDUCTION_BLOCK_SIZE >= 8)  { sum_vectors_v(sdata[tid], sdata[tid +  4], sdata[tid], dim1, dim2); }
        if (REDUCTION_BLOCK_SIZE >= 4)  { sum_vectors_v(sdata[tid], sdata[tid +  2], sdata[tid], dim1, dim2); }
        if (REDUCTION_BLOCK_SIZE >= 2)  { sum_vectors_v(sdata[tid], sdata[tid +  1], sdata[tid], dim1, dim2); }
    }

    if (tid == 0)
        deinit(sdata[0], reduction_buffer);
}


template<
    size_t dim1,
    size_t dim2,
    typename FInit,
    typename PixelPointerType
>
__global__
void init_reduction_step(
        PixelPointerType image,
        size_t width,
        uint32_t *samples,
        size_t n_classes,
        float *reduction_buffer,
        FInit init
) 
{
    size_t start = 0;

    for (size_t current_class = 0; current_class < n_classes; ++current_class)
    {
        reduce<dim1, dim2>(
            image, 
            width, 
            samples,
            current_class,
            start,
            reduction_buffer,
            init,
            Deinit<dim1, dim2>()
        );

        start += 1 + 2*samples[start];
    }
}


template <size_t dim1, size_t dim2>
void init_completion_step(
    CudaMemory<float> &reduction_buffer_d, 
    size_t n_classes, 
    float **constant_memory_d,
    float *cov_matrices_norms_d=nullptr
)
{
    const size_t reduction_buffer_offset = dim1 * dim2 * REDUCTION_GRID_SIZE;
    
    float constant_memory_h[MAX_N_CLASSES][dim1 * dim2];
    std::vector<float> reduction_buffer_h(reduction_buffer_d.count);

    reduction_buffer_d.memcpy(
        reduction_buffer_h.data(),
        cudaMemcpyDeviceToHost
    );
    
    for (size_t current_class = 0, grid_start = 0; current_class < n_classes; ++current_class, grid_start += reduction_buffer_offset)
    {
        for (size_t i = 0; i < dim1 * dim2; ++i)
            constant_memory_h[current_class][i] = 0;
        
        for (size_t vector_start = grid_start; vector_start < grid_start + reduction_buffer_offset; vector_start += dim1 * dim2)
            sum_vectors(
                reduction_buffer_h.data() + vector_start,
                constant_memory_h[current_class],
                constant_memory_h[current_class],
                dim1,
                dim2
            );
    }

    if (cov_matrices_norms_d != nullptr)
    {
        float cov_matrices_norms_h[MAX_N_CLASSES];

        for (size_t current_class = 0; current_class < n_classes; ++current_class)
        {
            cov_matrices_norms_h[current_class] = matrix_norm(
                constant_memory_h[current_class], 
                dim1, 
                dim2
            );

            inverse_matrix(
                constant_memory_h[current_class],
                constant_memory_h[current_class],
                dim1
            );
        }

        checkCudaErrors(cudaMemcpyToSymbol(
            cov_matrices_norms_d,
            cov_matrices_norms_h,
            MAX_N_CLASSES *  sizeof(float),
            0,
            cudaMemcpyHostToDevice
        ));
    }

    checkCudaErrors(cudaMemcpyToSymbol(
        constant_memory_d,
        constant_memory_h,
        MAX_N_CLASSES * dim1 * dim2 * sizeof(float),
        0,
        cudaMemcpyHostToDevice
    ));
}


template <
    size_t dim1,
    size_t dim2,
    typename PixelPointerType
>
void init_constant_memory(
    PixelPointerType input_image,
    size_t width,
    uint32_t *samples,
    size_t n_classes
)
{
    CudaMemory<float> reduction_buffer(DIM1 * DIM1 * REDUCTION_GRID_SIZE * MAX_N_CLASSES);

    init_reduction_step<dim1, dim2><<<REDUCTION_GRID_SIZE, REDUCTION_BLOCK_SIZE>>>(
        input_image,
        width,
        samples,
        n_classes,
        reduction_buffer.get(),
        InitAvg<dim1, dim2>()
    );

    init_completion_step<dim1, dim2>(
        reduction_buffer,
        n_classes,
        reinterpret_cast<float**>(avg)
    );

    init_reduction_step<dim1, dim1><<<REDUCTION_GRID_SIZE, REDUCTION_BLOCK_SIZE>>>(
        input_image,
        width,
        samples,
        n_classes,
        reduction_buffer.get(),
        InitCov<dim1, dim2>()
    );

    init_completion_step<dim1, dim1>(
        reduction_buffer,
        n_classes,
        reinterpret_cast<float**>(inversed_cov_matrices),
        reinterpret_cast<float*>(cov_matrices_norms)
    );
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

    CudaMemory<uchar4>  input_image_d(input_image_h.count()), 
                       output_image_d(input_image_h.count());

    input_image_d.memcpy(
        input_image_h.buffer.data(),
        cudaMemcpyHostToDevice
    );

    cudaError_t err = cudaSuccess;

    init_constant_memory<DIM1, DIM2>(
        input_image_d.get(),
        input_image_h.width,
        samples_d.get(),
        n_classes
    );

    if (err != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch kernel (error "
                  << cudaGetErrorString(err)
                  << ")!"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    Image<uchar4> output_image_h = input_image_h;

    output_image_d.memcpy(
        output_image_h.buffer.data(),
        cudaMemcpyDeviceToHost
    );

    output_image_h.save(output_name);
    
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
