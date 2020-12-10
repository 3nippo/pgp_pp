#pragma once

#include <vector>

#include "./dummy_helper.cuh"

class Lab05
{
public:
    using uint = unsigned int;        

private:
    uint n;

    std::vector<uint> numbers;
    std::vector<CudaMemory<uint>> prefixMemory;

    CudaTimer timer;

public:
    Lab05() {}

    void ReadInput();

    void InitGPUMemory();

    void Sort();

    void PrintResult();

    void PrintTextResult();

private:
    void GetBlocksNumAndLastBlockCount(uint count, uint *blocksNum, uint *lastBlockCount);
    
    void Scan(uint depth=0);

    uint GetUpperPowerOfTwo(uint v);

    uint GetPaddedCount(uint count);
};
