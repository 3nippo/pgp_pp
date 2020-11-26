#include "lab09cpu.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <iterator>

#include <cstdio>
#include <cstring>

template <typename T>
void read_in_container(T &container, size_t n=0)
{
    static_assert(
        std::is_same<
            typename std::iterator_traits<typename T::iterator>::iterator_category,
            std::random_access_iterator_tag
        >::value,
        "Wrong container"
    );

    if (n == 0)
        n = container.size();

    for (typename T::iterator it = container.begin(); n && it != container.end(); ++it, --n)
    {
        typename T::value_type a;

        std::cin >> a;

        *it = a;
    }
}

Lab09cpu::Lab09cpu(int argc, char **argv)
{
    init();

    shape[0] = block_shape[0] * process_grid_shape[0];
    shape[1] = block_shape[1] * process_grid_shape[1];
    shape[2] = block_shape[2] * process_grid_shape[2];
}

double& Lab09cpu::locate(long long i, long long j, long long k)
{
    return block[i * shape[1] * shape[2] + j * shape[2] + k];
}

double& Lab09cpu::locate(std::vector<double> &v, long long i, long long j, long long k)
{
    return v[i * shape[1] * shape[2] + j * shape[2] + k];
}

void Lab09cpu::init()
{
    // input
    
    read_in_container(process_grid_shape);

    read_in_container(block_shape);

    std::cin >> output_name;

    std::cin >> eps;

    read_in_container(l);

    std::cin >> boundaries.down
             >> boundaries.up
             >> boundaries.left
             >> boundaries.right
             >> boundaries.front
             >> boundaries.back;
    
    std::cin >> u_0;

    // input done
}

void Lab09cpu::solve() 
{
    block = std::vector<double>(shape[0] * shape[1] * shape[2], u_0);

    std::vector<double> prev_block = block;

    double n_x = block_shape[0] * process_grid_shape[0],
           n_y = block_shape[1] * process_grid_shape[1],
           n_z = block_shape[2] * process_grid_shape[2];

    double h_x_pow_minus_2 = n_x * n_x / l[0] / l[0],
           h_y_pow_minus_2 = n_y * n_y / l[1] / l[1],
           h_z_pow_minus_2 = n_z * n_z / l[2] / l[2],
           denominator = 2 * (h_x_pow_minus_2 + h_y_pow_minus_2 + h_z_pow_minus_2);
                                       
    while (true)
    {
        double max_abs_difference = 0;

        #pragma omp parallel for \
            schedule(static) \
            collapse(3) \
            reduction(max : max_abs_difference)
        for (long long i = 0; i < shape[0]; ++i)
            for (long long j = 0; j < shape[1]; ++j)
                for (long long k = 0; k < shape[2]; ++k)
                {
                    double u_left  = i == 0            ? boundaries.left  : locate(prev_block, i - 1, j, k),
                           u_right = i == shape[0] - 1 ? boundaries.right : locate(prev_block, i + 1, j, k),
                           u_front = j == 0            ? boundaries.front : locate(prev_block, i, j - 1, k),
                           u_back  = j == shape[1] - 1 ? boundaries.back  : locate(prev_block, i, j + 1, k),
                           u_down  = k == 0            ? boundaries.down  : locate(prev_block, i, j, k - 1),
                           u_up    = k == shape[2] - 1 ? boundaries.up    : locate(prev_block, i, j, k + 1);
                    
                    locate(i, j, k) =  (u_left  + u_right) * h_x_pow_minus_2;

                    locate(i, j, k) += (u_front + u_back ) * h_y_pow_minus_2;

                    locate(i, j, k) += (u_down  + u_up   ) * h_z_pow_minus_2;

                    locate(i, j, k) /= denominator;

                    max_abs_difference = std::max(std::abs(locate(i, j, k) - locate(prev_block, i, j, k)), max_abs_difference);
                }   
        
        if (max_abs_difference < eps)
            break;

        prev_block = block; 
    }
}

void Lab09cpu::write_answer()
{
    std::ofstream output(output_name);

    output << std::scientific << std::setprecision(6);

    const int value_buffer_size = 20;

    char value_buffer[value_buffer_size];
    memset(value_buffer, 0, value_buffer_size);

    for (long long k = 0; k < shape[2]; ++k)
        for (long long j = 0; j < shape[1]; ++j)
            for (long long i = 0; i < shape[0]; ++i)
            {
                output << locate(i, j, k);

                if (i != process_grid_shape[0] * block_shape[0] - 1)
                {
                    output << " ";
                }
                else
                {
                    output << "\n";
                    
                    if (j == process_grid_shape[1] * block_shape[1] - 1 && k != process_grid_shape[2] * block_shape[2] - 1)
                    {
                        output << "\n";
                    }
                }
            }
}
