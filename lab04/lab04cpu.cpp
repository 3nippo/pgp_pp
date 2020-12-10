#include "lab04cpu.hpp"

#define GRID_SIZE 4
#define BLOCK_SIZE 32

#define GRID_SIZE_dim3 dim3(GRID_SIZE, GRID_SIZE, 1)
#define BLOCK_SIZE_dim3 dim3(BLOCK_SIZE, BLOCK_SIZE, 1)

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <iterator>

#define loc(v, i, j) v[(j) * n + i]
#define index(i, j) ((j) * n + i)
#define locX(i, j) X[(j) * m + i]


void Lab04cpu::ReadInput()
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

void Lab04cpu::NullColumnDown(
    double *matrix,
    uint n,
    uint m,
    uint row,
    uint column
)
{
    double divisor = loc(matrix, row, column);

    for (uint j = column + 1; j < m; ++j)
        for (uint i = row + 1; i < n; ++i)
        {
            loc(matrix, i, j) -= loc(matrix, i, column) / divisor * loc(matrix, row, j);
        }
}


void Lab04cpu::NullColumnUp(
    double *matrix,
    uint n,
    uint m,
    uint k,
    uint row,
    uint column
)
{
    double divisor = loc(matrix, row, column);

    for (uint j = m; j < m + k; ++j)
        for (uint i = 0; i < row; ++i)
        {
            loc(matrix, i, j) -= loc(matrix, i, column) / divisor * loc(matrix, row, j);
        }
}

void Lab04cpu::Swap(
    double *matrix,
    uint n,
    uint m,
    uint column,
    uint lhs,
    uint rhs
)
{
    for (uint i = column; i < m; ++i)
    {
        double tmp = loc(matrix, lhs, i);
        loc(matrix, lhs, i) = loc(matrix, rhs, i);
        loc(matrix, rhs, i) = tmp;
    }
}

void Lab04cpu::ForwardGaussStroke()
{
    for (uint i = 0, j = 0; i < n && j < m; )
    {
        auto columnStart_d = matrix_h.begin(), columnEnd_d = matrix_h.begin(), ptrMax_d = columnEnd_d;

        std::advance(columnStart_d, index(i, j));
        std::advance(columnEnd_d, index(n, j));
        
        ptrMax_d = std::max_element(
            columnStart_d, 
            columnEnd_d, 
            [](const double a, const double b) 
            { 
                return std::abs(a) < std::abs(b); 
            }
        );

        if (std::abs(*ptrMax_d) <= zero)
        {
            ++j;
            continue;
        }

        stairIndexes.push_back({ i, j });

        if (i == n - 1)
        {
            break;
        }

        uint mainElementIndex = static_cast<uint>(std::distance(columnStart_d, ptrMax_d)) + i;
        
        if (mainElementIndex != i)
        {
            Swap(
                matrix_h.data(),
                n,
                m + k,
                j,
                i,
                mainElementIndex
            );
        }

        NullColumnDown(
            matrix_h.data(),
            n,
            m + k,
            i,
            j
        );

        ++i, ++j;
    }
}

void Lab04cpu::BackwardGaussStroke()
{
    for (uint i = stairIndexes.size() - 1; i + 1 > 0; --i)
    {
        NullColumnUp(
            matrix_h.data(),
            n,
            m,
            k,
            stairIndexes[i].first,
            stairIndexes[i].second
        );
    }

    for (uint p = 0; p < k; ++p)
    {
        for (uint i = stairIndexes.size() - 1; i + 1 > 0; --i)
        {
            locX(stairIndexes[i].second, p) = loc(matrix_h, stairIndexes[i].first, m + p) / loc(matrix_h, stairIndexes[i].first, stairIndexes[i].second);
        }
    }
}

void Lab04cpu::PrintX()
{
    for (uint i = 0; i < m; ++i)
    {
        for (uint j = 0; j < k; ++j)
            printf("%.10e ", locX(i, j));
        
        printf("\n");
    }
}

#undef loc
#undef index
#undef locX
