#include "lab04.cuh"

#define GRID_SIZE 128
#define BLOCK_SIZE 32

#define GRID_SIZE_dim3 dim3(GRID_SIZE, GRID_SIZE, 1)
#define BLOCK_SIZE_dim3 dim3(BLOCK_SIZE, BLOCK_SIZE, 1)

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#define loc(v, i, j) v[(j) * n + i]
#define index(i, j) ((j) * n + i)
#define locX(i, j) X[(j) * m + i]


void Lab04::ReadInput()
{
    scanf("%u%u%u", &n, &m, &k);

    matrix_h.resize(n * m + n * k);

    X.assign(m * k, 0);

    for (uint i = 0; i < n; ++i)
        for (uint j = 0; j < m; ++j)
        {
            double a;
            scanf("%lf", &a);

            loc(matrix_h, i, j) = a;
        }

    for (uint i = 0; i < n; ++i)
        for (uint j = 0; j < k; ++j)
        {
            double a;
            scanf("%lf", &a);

            loc(matrix_h, i, j + m) = a;
        }
}

void Lab04::InitGPUMemory()
{
    matrix_d.alloc(matrix_h.size());

    matrix_d.memcpy(matrix_h.data(), cudaMemcpyHostToDevice);

    timer.start();
}

__global__
void nullColumnDown(
    double *matrix,
    uint n,
    uint m,
    uint row,
    uint column
)
{
    const uint offsetX = blockDim.x * gridDim.x,
              offsetY = blockDim.y * gridDim.y;
    
    double divisor = loc(matrix, row, column);

    for (uint j = column + 1 + blockDim.y * blockIdx.y + threadIdx.y; j < m; j += offsetY)
        for (uint i = row + 1 + blockDim.x * blockIdx.x + threadIdx.x; i < n; i += offsetX)
        {
            loc(matrix, i, j) -= loc(matrix, i, column) / divisor * loc(matrix, row, j);
        }
}

__global__
void nullColumnUp(
    double *matrix,
    uint n,
    uint m,
    uint k,
    uint row,
    uint column
)
{
    const uint offsetX = blockDim.x * gridDim.x,
              offsetY = blockDim.y * gridDim.y;
    
    double divisor = loc(matrix, row, column);

    for (uint j = m + blockDim.y * blockIdx.y + threadIdx.y; j < m + k; j += offsetY)
        for (uint i = blockDim.x * blockIdx.x + threadIdx.x; i < row; i += offsetX)
        {
            loc(matrix, i, j) -= loc(matrix, i, column) / divisor * loc(matrix, row, j);
        }
}

__global__
void swap(
    double *matrix,
    uint n,
    uint m,
    uint column,
    uint lhs,
    uint rhs
)
{
    const uint offsetX = blockDim.x * gridDim.x;

    for (uint i = blockDim.x * blockIdx.x + threadIdx.x + column; i < m; i += offsetX)
    {
        double tmp = loc(matrix, lhs, i);
        loc(matrix, lhs, i) = loc(matrix, rhs, i);
        loc(matrix, rhs, i) = tmp;
    }
}

void Lab04::ForwardGaussStroke()
{
    thrust::device_ptr<double> ptrMatrix_d = thrust::device_pointer_cast(matrix_d.get());

    Cmp cmp;

    for (uint i = 0, j = 0; i < n && j < m; )
    {
        const thrust::device_ptr<double>  columnStart_d = ptrMatrix_d + index(i, j),
                                          columnEnd_d   = ptrMatrix_d + index(n, j),
                                          ptrMax_d      = thrust::max_element(columnStart_d, columnEnd_d, cmp);
        
        double maxx;

        checkCudaErrors(cudaMemcpy(
            &maxx,
            thrust::raw_pointer_cast(ptrMax_d),
            sizeof(double),
            cudaMemcpyDeviceToHost
        ));

        if (abs(maxx) <= zero)
        {
            ++j;
            continue;
        }

        stairIndexes.push_back({ i, j });

        CudaKernelChecker cudaKernelChecker;

        if (i == n - 1)
        {
            break;
        }

        uint mainElementIndex = static_cast<uint>(thrust::distance(columnStart_d, ptrMax_d)) + i;
        
        if (mainElementIndex != i)
        {
            swap<<<GRID_SIZE * GRID_SIZE, BLOCK_SIZE * BLOCK_SIZE>>>(
                matrix_d.get(),
                n,
                m + k,
                j,
                i,
                mainElementIndex
            );
            
            cudaKernelChecker.check("swap");
        }

        nullColumnDown<<<GRID_SIZE_dim3, BLOCK_SIZE_dim3>>>(
            matrix_d.get(),
            n,
            m + k,
            i,
            j
        );

        cudaKernelChecker.check("nullColumnDown");

        ++i, ++j;
    }
}

void Lab04::BackwardGaussStroke()
{
    for (uint i = stairIndexes.size() - 1; i + 1 > 0; --i)
    {
        CudaKernelChecker cudaKernelChecker;

        nullColumnUp<<<GRID_SIZE_dim3, BLOCK_SIZE_dim3>>>(
            matrix_d.get(),
            n,
            m,
            k,
            stairIndexes[i].first,
            stairIndexes[i].second
        );

        cudaKernelChecker.check("nullColumnUp");
    }

    matrix_d.memcpy(
        matrix_h.data(), 
        cudaMemcpyDeviceToHost
    );

    for (uint p = 0; p < k; ++p)
    {
        for (uint i = stairIndexes.size() - 1; i + 1 > 0; --i)
        {
            locX(stairIndexes[i].second, p) = loc(matrix_h, stairIndexes[i].first, m + p) / loc(matrix_h, stairIndexes[i].first, stairIndexes[i].second);
        }
    }
}

void Lab04::PrintX()
{
    timer.stop();

    for (uint i = 0; i < m; ++i)
    {
        for (uint j = 0; j < k; ++j)
            printf("%.10e ", locX(i, j));
        
        printf("\n");
    }

    timer.print_time();
}

#undef loc
#undef index
#undef locX
