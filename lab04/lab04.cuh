#pragma once

#include <vector>

#include "./dummy_helper.cuh"

typedef unsigned int uint;

class Lab04
{
public:
    struct Cmp
    {
        __host__ __device__
        bool operator()(const double lhs, const double rhs)
        {
            return abs(lhs) < abs(rhs);
        }
    };

private:
    const double zero = 1e-7;
    const uint precision = 10;

    uint n, m, k;

    std::vector<std::pair<uint, uint>> stairIndexes;
    
    std::vector<double> matrix_h;
    CudaMemory<double> matrix_d;

    std::vector<double> X;

    CudaTimer timer;

public:
    Lab04() {}

    void ReadInput();
    
    void InitGPUMemory();

    void ForwardGaussStroke();
    
    void BackwardGaussStroke();

    void PrintX();
};
