#include <iostream>
#include <iomanip>
#include <vector>

#include "dummy_helper.hpp"

#include <cuda_runtime.h>


__global__
void vector_max(const double *a, const double *b, double *result, size_t n)
{
    size_t offset = gridDim.x * blockDim.x;

    for (size_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < n; idx += offset)
        result[idx] = MAX(a[idx], b[idx]);
}


int main()
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    
    const size_t digits_after_point = 10;

    std::cout << std::scientific
              << std::setprecision(digits_after_point);


    size_t n;
    std::cin >> n;
    
    std::vector<double> a_h = read_vector1d<double>(n),
                       b_h = read_vector1d<double>(n),
                       result_h(n);

    CudaMemory<double> a_d(n), b_d(n), result_d(n);

    a_d.memcpy(a_h.data(), cudaMemcpyHostToDevice);
    b_d.memcpy(b_h.data(), cudaMemcpyHostToDevice);

    size_t threads_per_block = 256,
           blocks_per_grid = 16;
           /* blocks_per_grid = (n + threads_per_block - 1) / threads_per_block; */

    cudaError_t err = cudaSuccess;

    vector_max<<<blocks_per_grid, threads_per_block>>>(
        a_d.get(),
        b_d.get(),
        result_d.get(),
        n
    );

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vector_max kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    result_d.memcpy(result_h.data(), cudaMemcpyDeviceToHost);
      
    print_vector1d(result_h);

    return 0;
}
