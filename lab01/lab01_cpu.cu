#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include "dummy_helper.cuh"

#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define GRID_SIZE 256


void vector_max(const double *a, const double *b, double *result, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        result[i] = std::max(a[i], b[i]);
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

    CudaTimer timer;

    timer.start();

    vector_max(
        a_h.data(),
        b_h.data(),
        result_h.data(),
        n
    );

    timer.stop();
    timer.print_time();

    print_vector1d(result_h);

    return 0;
}
