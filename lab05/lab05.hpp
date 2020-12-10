#pragma once

#include <vector>

class Lab05
{
public:
    using uint = unsigned int;        

private:
    uint n;

    std::vector<uint> numbers;
    std::vector<std::vector<uint>> prefixMemory;

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

    void ScanBlockDown(
        uint *block,
        uint size,
        uint *blockTail
    );

    void ScanBlocksDown(
        uint *blocks,
        uint *tails,
        uint blocksNum,
        uint lastBlockCount
    );

    void ScanBlockUp(
        uint *block,
        uint size,
        uint toAdd
    );

    void ScanBlocksUp(
        uint *blocks,
        uint *tails,
        uint blocksNum,
        uint lastBlockCount
    );

    void NullVector(
        uint *v,
        uint count
    );

    void InitBitScan(
        uint *numbers,        
        uint *bits,
        uint bitNum,
        uint *prefixMemoryLvl0,        
        uint count
    );

    void PlaceNumbersByScan(
        uint *numbers,        
        uint *newNumbers,        
        uint *bits,        
        uint *scan,        
        uint count
    );
};
