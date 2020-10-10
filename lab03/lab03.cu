#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <limits>

#include "dummy_helper.cuh"
#include "image.hpp"
#include "math.cu"
#include <cmath>

#define MAX_N_CLASSES 32
#define REDUCTION_BLOCK_SIZE 256
#define REDUCTION_GRID_SIZE 16
#define BLOCK_SIZE 256
#define GRID_SIZE 16
#define DIM1 3
#define DIM2 1

__device__ __constant__ float avg[MAX_N_CLASSES][DIM1 * DIM2];
__constant__ float cov_matrices_norms[MAX_N_CLASSES];
__constant__ float inverse_cov_matrices[MAX_N_CLASSES][DIM1 * DIM1];


class PixelReader
{
public:
    template <typename PixelType>
    __host__ __device__
    static void read_pixel(float *v, PixelType pixel);
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
class AvgBuilder
{
public:
    __host__ __device__
    void operator()(float *a, size_t current_class, float *result)
    {
        for (size_t i = 0; i < dim1 * dim2; ++i)
            result[i] = a[i];
    }
};


template<
    size_t dim1,
    size_t dim2
>
class CovMatrixBuilder
{
public:
    __host__ __device__
    void operator()(float *a, size_t current_class, float *result)
    {
        float difference_with_avg[dim1*dim2];

        sum_vectors(a, avg[current_class], difference_with_avg, dim1, dim2);

        mul_vectors(
            difference_with_avg, 
            difference_with_avg,
            result,
            dim1,
            dim2,
            dim1
        );
    }
};


template <
    size_t dim1,
    size_t dim2,
    bool subtract,
    typename BuilderType
>
class Init
{
private:
    BuilderType builder;

public:
    __host__ __device__
    void operator()(
        float *a,
        size_t current_class,
        float *sdata,
        size_t size
    )
    {
        float ma[dim1*dim2];

        builder(a, current_class, ma);

        for (size_t i = 0; i < dim1*dim2; ++i)
            ma[i] /= (size - subtract);
        
        sum_vectors(sdata, ma, sdata, dim1, dim2);
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

    const size_t tid = threadIdx.x;
    
    for (size_t i = 0; i < dim1 * dim2; ++i)
        sdata[tid][i] = 0;

    const size_t combined_block_size = REDUCTION_BLOCK_SIZE * 2; // 2 for each pixel and 2 is the size of combination
    
    const size_t grid_size = REDUCTION_GRID_SIZE * combined_block_size;
    
    const size_t sample_end = start + 1 + 2 * samples[start];

    for (
            size_t pixel_sample_position = start + 1 + blockIdx.x * combined_block_size + 2 * tid; 
            pixel_sample_position < sample_end; 
            pixel_sample_position += grid_size
        ) 
    { 
        float a[dim1*dim2];
        size_t position = samples[pixel_sample_position + 1] * width + samples[pixel_sample_position];

        PixelReader::read_pixel(a, image[position]);

        init(
            a,
            current_class,
            sdata[tid],
            samples[start]
        );

        /* if (pixel_sample_position + combined_block_size / 2 < sample_end) */
        /* { */
        /*     float b[dim1*dim2]; */
        /*     size_t next_position = samples[pixel_sample_position + 1 + combined_block_size / 2] * width + samples[pixel_sample_position + combined_block_size / 2]; */

        /*     PixelReader::read_pixel(b, image[next_position]); */

        /*     init( */
        /*         b, */
        /*         current_class, */
        /*         sdata[tid] */
        /*     ); */
        /* } */
    }
    
    __syncthreads();
    
    if (REDUCTION_BLOCK_SIZE >= 512) { if (tid < 256) { sum_vectors_v(sdata[tid], sdata[tid + 256], sdata[tid], dim1, dim2); } __syncthreads(); }
    if (REDUCTION_BLOCK_SIZE >= 256) { if (tid < 128) { sum_vectors_v(sdata[tid], sdata[tid + 128], sdata[tid], dim1, dim2); } __syncthreads(); }
    if (REDUCTION_BLOCK_SIZE >= 128) { if (tid < 64)  { sum_vectors_v(sdata[tid], sdata[tid +  64], sdata[tid], dim1, dim2); } __syncthreads(); }
    if (REDUCTION_BLOCK_SIZE >=  64) { if (tid < 32)  { sum_vectors_v(sdata[tid], sdata[tid +  32], sdata[tid], dim1, dim2); } __syncthreads(); }
    if (REDUCTION_BLOCK_SIZE >=  32) { if (tid < 32)  { sum_vectors_v(sdata[tid], sdata[tid +  16], sdata[tid], dim1, dim2); } __syncthreads(); }
    if (REDUCTION_BLOCK_SIZE >=  16) { if (tid < 32)  { sum_vectors_v(sdata[tid], sdata[tid +   8], sdata[tid], dim1, dim2); } __syncthreads(); }
    if (REDUCTION_BLOCK_SIZE >=   8) { if (tid < 32)  { sum_vectors_v(sdata[tid], sdata[tid +   4], sdata[tid], dim1, dim2); } __syncthreads(); }
    if (REDUCTION_BLOCK_SIZE >=   4) { if (tid < 32)  { sum_vectors_v(sdata[tid], sdata[tid +   2], sdata[tid], dim1, dim2); } __syncthreads(); }
    if (REDUCTION_BLOCK_SIZE >=   2) { if (tid < 32)  { sum_vectors_v(sdata[tid], sdata[tid +   1], sdata[tid], dim1, dim2); } __syncthreads(); }
    
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
    size_t reduction_buffer_offset = dim1 * dim2 * REDUCTION_GRID_SIZE;

    for (size_t current_class = 0; current_class < n_classes; ++current_class)
    {
        reduce<dim1, dim2>(
            image, 
            width, 
            samples,
            current_class,
            start,
            reduction_buffer + reduction_buffer_offset * current_class,
            init,
            Deinit<dim1, dim2>()
        );

        start += 1 + 2*samples[start];
    }
}


template <size_t dim1, size_t dim2, bool is_avg>
void init_completion_step(
    CudaMemory<float> &reduction_buffer_d, 
    std::vector<uint32_t> &samples,
    const size_t n_classes
)
{
    const size_t reduction_buffer_offset = dim1 * dim2 * REDUCTION_GRID_SIZE;
    
    float constant_memory_h[MAX_N_CLASSES][dim1 * dim2];
    std::vector<float> reduction_buffer_h(reduction_buffer_d.count);

    reduction_buffer_d.memcpy(
        reduction_buffer_h.data(),
        cudaMemcpyDeviceToHost
    );

    for (
            size_t current_class = 0, grid_start = 0, sample_start = 0; 
            current_class < n_classes; 
            ++current_class, grid_start += reduction_buffer_offset, sample_start += 1 + 2 * samples[sample_start]
        )
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
        
        /* bool subtract = !is_avg; */
        /* for (size_t i = 0; i < dim1 * dim2; ++i) */
        /*     constant_memory_h[current_class][i] /= (samples[sample_start] - subtract); */
        
        if (is_avg)
            revert_sign(constant_memory_h[current_class], dim1, dim2);
    }

    if (avg)
    {
        for (size_t i = 0; i < n_classes; ++i)
        {
            for (size_t j = 0; j < dim1*dim2; ++j)
                std::cout << constant_memory_h[i][j] << ' ';
            std::cout << std::endl;
        }
                
    }

    if (!is_avg)
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

        for (size_t i = 0; i < n_classes; ++i)
        {
            for (size_t j = 0; j < dim1*dim2; ++j)
                std::cout << constant_memory_h[i][j] << ' ';
            std::cout << std::endl;
        }

        checkCudaErrors(cudaMemcpyToSymbol(
            cov_matrices_norms,
            cov_matrices_norms_h,
            MAX_N_CLASSES *  sizeof(float),
            0,
            cudaMemcpyHostToDevice
        ));
        
        checkCudaErrors(cudaMemcpyToSymbol(
            inverse_cov_matrices,
            constant_memory_h,
            MAX_N_CLASSES * dim1 * dim2 * sizeof(float)
        ));
    }
    else
    {
        checkCudaErrors(cudaMemcpyToSymbol(
            avg,
            constant_memory_h,
            MAX_N_CLASSES * dim1 * dim2 * sizeof(float)
        ));
    }
}


template <
    size_t dim1,
    size_t dim2,
    typename PixelPointerType
>
void init_constant_memory(
    PixelPointerType input_image,
    const size_t width,
    std::vector<uint32_t> &samples_h,
    uint32_t *samples_d,
    const size_t n_classes
)
{
    CudaMemory<float> reduction_buffer(dim1 * dim1 * REDUCTION_GRID_SIZE * MAX_N_CLASSES);
    
    CudaKernelChecker checker;

    init_reduction_step<dim1, dim2><<<REDUCTION_GRID_SIZE, REDUCTION_BLOCK_SIZE>>>(
        input_image,
        width,
        samples_d,
        n_classes,
        reduction_buffer.get(),
        Init<dim1, dim2, false, AvgBuilder<dim1, dim2>>()
    );

    checker.check("init_reduction_step<InitAvg>");

    init_completion_step<dim1, dim2, true>(
        reduction_buffer,
        samples_h,
        n_classes
    );

    init_reduction_step<dim1, dim1><<<REDUCTION_GRID_SIZE, REDUCTION_BLOCK_SIZE>>>(
        input_image,
        width,
        samples_d,
        n_classes,
        reduction_buffer.get(),
        Init<dim1, dim1, true, CovMatrixBuilder<dim1, dim2>>()
    );

    checker.check("init_reduction_step<InitCov>");

    init_completion_step<dim1, dim1, false>(
        reduction_buffer,
        samples_h,
        n_classes
    );
}


template <
    size_t dim1, 
    size_t dim2, 
    typename PixelPointerType
>
__global__
void kernel(
    PixelPointerType input_image,
    const size_t image_buffer_count,
    const size_t n_classes,
    const float float_lowest
)
{
    size_t offset = blockDim.x * gridDim.x;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < image_buffer_count; i += offset)
    {
        float mmp_max = float_lowest;
        size_t mmp_max_class;

        PixelReader reader;
        float pixel[dim1 * dim2];
        
        reader.read_pixel(pixel, input_image[i]);

        for (size_t current_class = 0; current_class < n_classes; ++current_class)
        {    
            float difference_with_avg[dim1 * dim2];
            sum_vectors(
                pixel, 
                avg[current_class], 
                difference_with_avg, 
                dim1, 
                dim2
            );

            float mul_buffer[dim2 * dim1];
            mul_vectors(
                difference_with_avg, 
                inverse_cov_matrices[current_class],
                mul_buffer,
                dim2,
                dim1,
                dim1
            );

            float result;
            mul_vectors(
                mul_buffer,
                difference_with_avg,
                &result,
                dim2,
                dim1,
                dim2
            );

            result += std::log(cov_matrices_norms[current_class]);

            if (-result > mmp_max)
            {
                mmp_max = -result;
                mmp_max_class = current_class;
            }
        }

        input_image[i] = { 
            static_cast<unsigned char>(pixel[0]), 
            static_cast<unsigned char>(pixel[1]), 
            static_cast<unsigned char>(pixel[2]), 
            static_cast<unsigned char>(mmp_max_class) 
        };
    }
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

    CudaMemory<uchar4>  input_image_d(input_image_h.count());

    input_image_d.memcpy(
        input_image_h.buffer.data(),
        cudaMemcpyHostToDevice
    );

    init_constant_memory<DIM1, DIM2>(
        input_image_d.get(),
        input_image_h.width,
        samples_h,
        samples_d.get(),
        n_classes
    );

    CudaKernelChecker checker;

    kernel<DIM1, DIM2><<<GRID_SIZE, BLOCK_SIZE>>>(
        input_image_d.get(),
        input_image_h.count(),
        n_classes,
        std::numeric_limits<float>::lowest()
     );

    checker.check("kernel");

    Image<uchar4> output_image_h = input_image_h;

    input_image_d.memcpy(
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
        std::cout << "ERROR: " << err.what() << std::endl;
        return 1;
    }

    return 0;
}
