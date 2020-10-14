#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <limits>

#include "image.hpp"
#include "math.cu"
#include <cmath>
#include "dummy_helper.cuh"

#define DIM1 3
#define DIM2 1
#define MAX_N_CLASSES 32


std::vector<std::vector<float>> avg;
std::vector<float> cov_matrices_norms;
std::vector<std::vector<float>> inverse_cov_matrices;


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
    void operator()(float *a, size_t current_class, float *result)
    {
        float difference_with_avg[dim1*dim2];

        sum_vectors(a, avg[current_class].data(), difference_with_avg, dim1, dim2);

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


template <
    size_t dim1,
    size_t dim2,
    typename FInit,
    typename PixelPointerType
>
void summate(
        PixelPointerType image, 
        size_t width, 
        std::vector<uint32_t> &samples,
        size_t current_class,
        size_t start,
        FInit init,
        std::vector<std::vector<float>> &sdata
) 
{
    const size_t sample_end = start + 1 + 2 * samples[start];
    
    std::vector<std::vector<float>> fake_data(256, std::vector<float>(dim1*dim2));

    for (size_t blockidx = 0; blockidx < 16; ++blockidx)
    {
    for (size_t tid = 0; tid < 256; ++tid)
    for (
            size_t pixel_sample_position = start + 1 + blockidx * 256 + 2*tid; 
            pixel_sample_position < sample_end; 
            pixel_sample_position += 256 * 2 * 16
        ) 
    { 
        float a[dim1*dim2];
        size_t position = samples[pixel_sample_position + 1] * width + samples[pixel_sample_position];

        PixelReader::read_pixel(a, image[position]);

        init(
            a,
            current_class,
            fake_data[tid].data(),
            samples[start]
        );
    }
    for (size_t i = 0; i < fake_data.size(); ++i)
        sum_vectors(
            fake_data[i].data(),
            sdata[current_class].data(),
            sdata[current_class].data(),
            dim1,
            dim2
        );

    fake_data.assign(fake_data.size(), std::vector<float>(dim1*dim2));
    }
}


template<
    size_t dim1,
    size_t dim2,
    typename FInit,
    typename PixelPointerType
>
void init_reduction_step(
        PixelPointerType image,
        size_t width,
        std::vector<uint32_t> &samples,
        size_t n_classes,
        FInit init,
        std::vector<std::vector<float>> &v
) 
{
    size_t start = 0;

    for (size_t current_class = 0; current_class < n_classes; ++current_class)
    {
        summate<dim1, dim2>(
            image, 
            width, 
            samples,
            current_class,
            start,
            init,
            v
        );

        start += 1 + 2*samples[start];
    }
}


template <size_t dim1, size_t dim2, bool is_avg>
void init_completion_step(
    std::vector<std::vector<float>> &constant_memory_h,
    std::vector<uint32_t> &samples,
    const size_t n_classes
)
{
    for (
            size_t current_class = 0,  sample_start = 0; 
            current_class < n_classes; 
            ++current_class, sample_start += 1 + 2 * samples[sample_start]
        )
    {
        /* bool subtract = !is_avg; */
        /* for (size_t i = 0; i < dim1 * dim2; ++i) */
        /*     constant_memory_h[current_class][i] /= (samples[sample_start] - subtract); */
        
        if (is_avg)
            revert_sign(constant_memory_h[current_class].data(), dim1, dim2);
    }

    if (!is_avg)
    {
        for (size_t current_class = 0; current_class < n_classes; ++current_class)
        {
            cov_matrices_norms[current_class] = matrix_norm(
                constant_memory_h[current_class].data(), 
                dim1, 
                dim2
            );

            inverse_matrix(
                constant_memory_h[current_class].data(),
                constant_memory_h[current_class].data(),
                dim1
            );
        }
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
    const size_t n_classes
)
{
    init_reduction_step<dim1, dim2>(
        input_image,
        width,
        samples_h,
        n_classes,
        Init<dim1, dim2, false, AvgBuilder<dim1, dim2>>(),
        avg
    );

    init_completion_step<dim1, dim2, true>(
        avg,
        samples_h,
        n_classes
    );

    init_reduction_step<dim1, dim1>(
        input_image,
        width,
        samples_h,
        n_classes,
        Init<dim1, dim1, true, CovMatrixBuilder<dim1, dim2>>(),
        inverse_cov_matrices
    );

    init_completion_step<dim1, dim1, false>(
        inverse_cov_matrices,
        samples_h,
        n_classes
    );
}


template <
    size_t dim1, 
    size_t dim2, 
    typename PixelPointerType
>
void kernel(
    PixelPointerType input_image,
    const size_t image_buffer_count,
    const size_t n_classes,
    const float float_lowest
)
{
    for (size_t i = 0; i < image_buffer_count; ++i)
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
                avg[current_class].data(), 
                difference_with_avg, 
                dim1, 
                dim2
            );

            float mul_buffer[dim2 * dim1];
            mul_vectors(
                difference_with_avg, 
                inverse_cov_matrices[current_class].data(),
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
    avg = std::vector<std::vector<float>>(MAX_N_CLASSES, std::vector<float>(DIM1 * DIM2));
    
    cov_matrices_norms = std::vector<float>(MAX_N_CLASSES);
    
    inverse_cov_matrices = std::vector<std::vector<float>>(MAX_N_CLASSES, std::vector<float>(DIM1 * DIM1));

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
    
    Image<uchar4> input_image_h(input_name);

    init_constant_memory<DIM1, DIM2>(
        input_image_h.buffer.data(),
        input_image_h.width,
        samples_h,
        n_classes
    );

    /* for (size_t i = 0; i < n_classes; ++i) */
    /* { */
    /*     for (size_t j = 0; j < DIM1*DIM2; ++j) */
    /*         std::cout << avg[i][j] << ' '; */
    /*     std::cout << std::endl; */
    /* } */

    /* for (size_t i = 0; i < n_classes; ++i) */
    /* { */
    /*     for (size_t j = 0; j < DIM1*DIM1; ++j) */
    /*         std::cout << inverse_cov_matrices[i][j] << ' '; */
    /*     std::cout << std::endl; */
    /* } */
    
    CudaTimer timer;

    timer.start();
    
    kernel<DIM1, DIM2>(
        input_image_h.buffer.data(),
        input_image_h.count(),
        n_classes,
        std::numeric_limits<float>::lowest()
    );

    timer.stop();
    timer.print_time();

    Image<uchar4> output_image_h = input_image_h;

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
