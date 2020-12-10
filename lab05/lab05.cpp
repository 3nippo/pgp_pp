#include "./lab05.hpp"

#include <iostream>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cmath>

#define SHARED_MEMORY_SIZE ((1 << 13) + (1 << 8))
#define DATA_BLOCK_SIZE (1 << 13)
#define BLOCK_SIZE 512
#define GRID_SIZE 16

void Lab05::ReadInput()
{
    std::freopen(nullptr, "rb", stdin);

    if (std::ferror(stdin))
        throw std::runtime_error("Failed to reopen stdin");

    if (
        std::fread(&n, sizeof(n), 1, stdin) != 1
        || std::ferror(stdin) && !std::feof(stdin)    
    )
        throw std::runtime_error("Bad input");
    
    if (n == 0)
        return;

    numbers.assign(n, 0);

    if (
        std::fread(numbers.data(), sizeof(uint), n, stdin) != n
        || std::ferror(stdin) && !std::feof(stdin)    
    )
        throw std::runtime_error("Bad input");
}

void Lab05::ScanBlockDown(
    uint *block,
    uint size,
    uint *blockTail
)
{
    std::vector<uint> buffer(DATA_BLOCK_SIZE);

    if (size == 0)
        size = DATA_BLOCK_SIZE;

    for (uint i = 0; i < size; ++i)
        buffer[i] = block[i];

    // reduce down
    
    uint iterOffset = 1;
    
    for (uint d = size >> 1; d != 0; d >>= 1)
    {
        for (uint i = 0; i < d; ++i)
        {
            const uint ai = iterOffset * (2 * i + 1) - 1,
                       bi = iterOffset * (2 * i + 2) - 1;

            buffer[bi] += buffer[ai];
        }

        iterOffset <<= 1;
    }
    
    // save and null last element
    
    uint lastValue;

    lastValue = buffer[size - 1];
    buffer[size - 1] = 0;

    // reduce up
    
    for (uint d = 1; d < size; d <<= 1)
    {
        iterOffset >>= 1;
        
        if (iterOffset != 0)
            for (uint i = 0; i < d; ++i)
            {
                const uint ai = iterOffset * (2 * i + 1) - 1,
                           bi = iterOffset * (2 * i + 2) - 1;

                uint toAdd = buffer[ai];
                buffer[ai] = buffer[bi];
                buffer[bi] += toAdd;
            }
    }

    // write back result and tail
    
    for (uint i = 0; i < size - 1; ++i)
        block[i] = buffer[i + 1];

    block[size - 1] = lastValue;

    *blockTail = block[size - 1];
}

#undef index

void Lab05::ScanBlocksDown(
    uint *blocks,
    uint *tails,
    uint blocksNum,
    uint lastBlockCount
)
{
    uint i = 0;

    for (; i < blocksNum - 1; ++i)
    {
        ScanBlockDown(
            blocks + DATA_BLOCK_SIZE * i,
            0,
            tails + i
        );
    }
    
    ScanBlockDown(
        blocks + DATA_BLOCK_SIZE * i,
        lastBlockCount,
        tails + i
    );
}

void Lab05::ScanBlockUp(
    uint *block,
    uint size,
    uint toAdd
)
{
    if (size == 0)
        size = DATA_BLOCK_SIZE;
    
    for (uint i = 0; i < size; ++i)
        block[i] += toAdd;   
}

void Lab05::ScanBlocksUp(
    uint *blocks,
    uint *tails,
    uint blocksNum,
    uint lastBlockCount
)
{
    uint i = 0;

    for (; i < blocksNum - 1; ++i)
    {
        ScanBlockUp(
            blocks + DATA_BLOCK_SIZE * i,
            0,
            *(tails + i - 1)
        );
    }
    
    ScanBlockUp(
        blocks + DATA_BLOCK_SIZE * i,
        lastBlockCount,
        *(tails + i - 1)
    );
}

void Lab05::GetBlocksNumAndLastBlockCount(uint count, uint *blocksNum, uint *lastBlockCount)
{
    *lastBlockCount = count % DATA_BLOCK_SIZE;

    *blocksNum = count / DATA_BLOCK_SIZE + (*lastBlockCount != 0);
}

void Lab05::NullVector(
    uint *v,
    uint count
)
{
    uint i = 0;

    for (; i < count; ++i)
    {
        v[i] = 0;
    }
}

void Lab05::Scan(uint depth)
{   
    if (depth == prefixMemory.size() - 1)
        return;

    uint blocksNum, lastBlockCount;

    GetBlocksNumAndLastBlockCount(
        prefixMemory[depth].size(), 
        &blocksNum,
        &lastBlockCount
    );
    
    NullVector(
        prefixMemory[depth + 1].data(),
        prefixMemory[depth + 1].size()
    );

    ScanBlocksDown(
        prefixMemory[depth].data(),
        prefixMemory[depth + 1].data(),
        blocksNum,
        lastBlockCount == 0 ? DATA_BLOCK_SIZE : lastBlockCount
    );

    Scan(depth + 1);
    
    ScanBlocksUp(
        prefixMemory[depth].data(),
        prefixMemory[depth + 1].data(),
        blocksNum,
        lastBlockCount == 0 ? DATA_BLOCK_SIZE : lastBlockCount
    );
}

Lab05::uint Lab05::GetUpperPowerOfTwo(uint v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    
    return v;
}

Lab05::uint Lab05::GetPaddedCount(uint count)
{
    const uint tailCount = count % DATA_BLOCK_SIZE;

    return count / DATA_BLOCK_SIZE * DATA_BLOCK_SIZE + GetUpperPowerOfTwo(tailCount);
}

void Lab05::InitGPUMemory()
{
    if (n < 2)
        return;

    std::vector<uint> workObj;

    uint currentCount = GetPaddedCount(n);
    
    prefixMemory.reserve(2 * std::ceil(std::log(n) / std::log(DATA_BLOCK_SIZE)));

    while (true)
    {
        uint blocksNum, lastBlockCount;
        
        GetBlocksNumAndLastBlockCount(
            currentCount, 
            &blocksNum,
            &lastBlockCount
        );

        workObj.resize(currentCount);

        prefixMemory.push_back(std::move(workObj));

        if (blocksNum == 1)
            break;

        currentCount = GetPaddedCount(blocksNum);
    }

    workObj.resize(1);

    prefixMemory.push_back(std::move(workObj));
}

void Lab05::InitBitScan(
    uint *numbers,        
    uint *bits,
    uint bitNum,
    uint *prefixMemoryLvl0,        
    uint count
)
{
    for (uint i = 0; i < count; ++i)
    {
        const uint number = numbers[i];

        bits[i] = (number >> bitNum) & 1;
        prefixMemoryLvl0[i] = (number >> bitNum) & 1;
    }
}

void Lab05::PlaceNumbersByScan(
    uint *numbers,        
    uint *newNumbers,        
    uint *bits,        
    uint *scan,        
    uint count
)
{
    uint i = 0;
    
    uint sn = scan[count - 1];
    
    if (bits[0] == 0)
        newNumbers[0] = numbers[0];
    else
        newNumbers[count - sn] = numbers[0];

    ++i;

    for (; i < count; ++i)
    {
        if (bits[i] == 0)
            newNumbers[i - scan[i - 1]] = numbers[i];
        else
            newNumbers[count - sn + scan[i - 1]] = numbers[i];
    }
}

void Lab05::Sort()
{
    if (n < 2)
        return;

    std::vector<uint> numbers_d(numbers.size()), 
                      newNumbers_d(numbers.size()),
                      bits(numbers.size());

    numbers_d = numbers;
    
    for (uint i = 0; i < 32; ++i)
    {
        NullVector(
            prefixMemory[0].data(),
            prefixMemory[0].size()
        );

        InitBitScan(
            numbers_d.data(),
            bits.data(),
            i,
            prefixMemory[0].data(),
            numbers_d.size()
        );

        Scan();
    
        PlaceNumbersByScan(
            numbers_d.data(),
            newNumbers_d.data(),
            bits.data(),
            prefixMemory[0].data(),
            numbers_d.size()
        );

        std::swap(numbers_d, newNumbers_d);
    }
    
    numbers = numbers_d;
}

void Lab05::PrintTextResult()
{
    /* std::vector<uint> data_h(data_d.count); */

    /* data_d.memcpy(data_h.data(), cudaMemcpyDeviceToHost); */
    
    for (const auto el : numbers)
        std::cout << el << ' ';
    std::cout << std::endl;
}

void Lab05::PrintResult()
{
    std::freopen(nullptr, "wb", stdout);

    if (std::ferror(stdout))
        throw std::runtime_error("Failed to reopen stdout");

    if (
        std::fwrite(numbers.data(), sizeof(uint), n, stdout) != n
        || std::ferror(stdout)
    )
        throw std::runtime_error("Failed to write to stdout");
}
