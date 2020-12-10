#pragma once

#include <vector>

typedef unsigned int uint;

class Lab04cpu
{
private:
    const double zero = 1e-7;

    uint n, m, k;

    std::vector<std::pair<uint, uint>> stairIndexes;
    
    std::vector<double> matrix_h;

    std::vector<double> X;

public:
    Lab04cpu() {}

    void ReadInput();
    
    void ForwardGaussStroke();
    
    void BackwardGaussStroke();

    void PrintX();

private:
    void NullColumnDown(
        double *matrix,
        uint n,
        uint m,
        uint row,
        uint column
    );

    void NullColumnUp(
        double *matrix,
        uint n,
        uint m,
        uint k,
        uint row,
        uint column
    );

    void Swap(
        double *matrix,
        uint n,
        uint m,
        uint column,
        uint lhs,
        uint rhs
    );
};
