#include "./lab05.cuh"

#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cmath>

#define SHARED_MEMORY_SIZE ((1 << 13) + (1 << 8))
#define DATA_BLOCK_SIZE (1 << 13)
#define BLOCK_SIZE 32
#define GRID_SIZE 32

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

#define index(i) (i + ((i) >> 5))

__device__
void scanBlockDown(
    Lab05::uint *block,
    Lab05::uint size,
    Lab05::uint *blockTail
)
{
    __shared__ Lab05::uint buffer[SHARED_MEMORY_SIZE];

    const Lab05::uint id     = threadIdx.x,
                      offset = blockDim.x;
    
    if (size == 0)
        size = DATA_BLOCK_SIZE;

    for (Lab05::uint i = id; i < size; i += offset)
        buffer[index(i)] = block[i];

    __syncthreads();
    
    // reduce down
    
    Lab05::uint iterOffset = 1;
    
    for (Lab05::uint d = size >> 1; d != 0; d >>= 1)
    {
        for (Lab05::uint i = id; i < d; i += offset)
        {
            const Lab05::uint ai = index(iterOffset * (2 * i + 1) - 1),
                              bi = index(iterOffset * (2 * i + 2) - 1);

            buffer[bi] += buffer[ai];
        }

        iterOffset <<= 1;

        __syncthreads();
    }
    
    // save and null last element
    
    Lab05::uint lastValue;

    if (id == 0)
    {
        lastValue = buffer[index(size - 1)];
        buffer[index(size - 1)] = 0;
    }

    __syncthreads();

    // reduce up
    
    for (Lab05::uint d = 1; d < size; d <<= 1)
    {
        iterOffset >>= 1;
        
        if (iterOffset != 0)
            for (Lab05::uint i = id; i < d; i += offset)
            {
                const Lab05::uint ai = index(iterOffset * (2 * i + 1) - 1),
                                  bi = index(iterOffset * (2 * i + 2) - 1);

                Lab05::uint toAdd = buffer[ai];
                buffer[ai] = buffer[bi];
                buffer[bi] += toAdd;
            }
        
        __syncthreads();
    }

    // write back result and tail
    
    for (Lab05::uint i = id; i < size - 1; i += offset)
        block[i] = buffer[index(i + 1)];

    if (id == 0)
    {
        block[size - 1] = lastValue;

        *blockTail = block[size - 1];
    }
}

#undef index

__global__
void scanBlocksDown(
    Lab05::uint *blocks,
    Lab05::uint *tails,
    Lab05::uint blocksNum,
    Lab05::uint lastBlockCount
)
{
    Lab05::uint i = blockIdx.x;

    for (; i < blocksNum - 1; i += gridDim.x)
    {
        scanBlockDown(
            blocks + DATA_BLOCK_SIZE * i,
            0,
            tails + i
        );
    }
    
    if (i == blocksNum - 1)
        scanBlockDown(
            blocks + DATA_BLOCK_SIZE * i,
            lastBlockCount,
            tails + i
        );
}

__device__
void scanBlockUp(
    Lab05::uint *block,
    Lab05::uint size,
    Lab05::uint toAdd
)
{
    if (size == 0)
        size = DATA_BLOCK_SIZE;
    
    for (Lab05::uint i = threadIdx.x; i < size; i += blockDim.x)
        block[i] += toAdd;   
}

__global__
void scanBlocksUp(
    Lab05::uint *blocks,
    Lab05::uint *tails,
    Lab05::uint blocksNum,
    Lab05::uint lastBlockCount
)
{
    Lab05::uint i = blockIdx.x + 1;

    for (; i < blocksNum - 1; i += gridDim.x)
    {
        scanBlockUp(
            blocks + DATA_BLOCK_SIZE * i,
            0,
            *(tails + i - 1)
        );
    }
    
    if (i == blocksNum - 1)
        scanBlockUp(
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

__global__
void nullVector(
    Lab05::uint *v,
    Lab05::uint count
)
{
    Lab05::uint i = blockDim.x * blockIdx.x + threadIdx.x;
    const Lab05::uint offset = blockDim.x * gridDim.x;

    for (; i < count; i += offset)
    {
        v[i] = 0;
    }
}

void Lab05::Scan(uint depth)
{   
    if (depth == prefixMemory.size() - 1)
        return;

    CudaKernelChecker cudaKernelChecker;
    
    uint blocksNum, lastBlockCount;

    GetBlocksNumAndLastBlockCount(
        prefixMemory[depth].count, 
        &blocksNum,
        &lastBlockCount
    );
    
    nullVector<<<GRID_SIZE, BLOCK_SIZE>>>(
        prefixMemory[depth + 1].get(),
        prefixMemory[depth + 1].count
    );

    cudaKernelChecker.check("nullVector");

    scanBlocksDown<<<GRID_SIZE, BLOCK_SIZE>>>(
        prefixMemory[depth].get(),
        prefixMemory[depth + 1].get(),
        blocksNum,
        lastBlockCount == 0 ? DATA_BLOCK_SIZE : lastBlockCount
    );

    cudaKernelChecker.check("scanBlocksDown");
    
    Scan(depth + 1);
    
    scanBlocksUp<<<GRID_SIZE, BLOCK_SIZE>>>(
        prefixMemory[depth].get(),
        prefixMemory[depth + 1].get(),
        blocksNum,
        lastBlockCount == 0 ? DATA_BLOCK_SIZE : lastBlockCount
    );

    cudaKernelChecker.check("scanBlocksUp");
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

    CudaMemory<uint> workObj;

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

        workObj.alloc(currentCount);

        prefixMemory.push_back(workObj);
        workObj.Reset();

        if (blocksNum == 1)
            break;

        currentCount = GetPaddedCount(blocksNum);
    }

    workObj.alloc(1);

    prefixMemory.push_back(workObj);
    workObj.Reset();

    timer.start();
}

__global__
void initBitScan(
    Lab05::uint *numbers,        
    Lab05::uint *bits,
    Lab05::uint bitNum,
    Lab05::uint *prefixMemoryLvl0,        
    Lab05::uint count
)
{
    const Lab05::uint id     = blockDim.x * blockIdx.x + threadIdx.x,
                      offset = blockDim.x * gridDim.x;

    for (Lab05::uint i = id; i < count; i += offset)
    {
        const Lab05::uint number = numbers[i];

        bits[i] = (number >> bitNum) & 1;
        prefixMemoryLvl0[i] = (number >> bitNum) & 1;
    }
}

__global__
void placeNumbersByScan(
    Lab05::uint *numbers,        
    Lab05::uint *newNumbers,        
    Lab05::uint *bits,        
    Lab05::uint *scan,        
    Lab05::uint count
)
{
    Lab05::uint     i = blockDim.x * blockIdx.x + threadIdx.x,
               offset = blockDim.x * gridDim.x;
    
    Lab05::uint sn = scan[count - 1];
    
    if (i == 0)
    {
        if (bits[0] == 0)
            newNumbers[0] = numbers[0];
        else
            newNumbers[count - sn] = numbers[0];

        i += offset;
    }

    for (; i < count; i += offset)
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

    CudaMemory<uint> numbers_d(numbers.size()), 
                     newNumbers_d(numbers.size()),
                     bits(numbers.size());

    numbers_d.memcpy(numbers.data(), cudaMemcpyHostToDevice);
    
    CudaKernelChecker cudaKernelChecker;

    for (uint i = 0; i < 32; ++i)
    {
        nullVector<<<GRID_SIZE, BLOCK_SIZE>>>(
            prefixMemory[0].get(),
            prefixMemory[0].count
        );

        cudaKernelChecker.check("nullVector");

        initBitScan<<<GRID_SIZE, BLOCK_SIZE>>>(
            numbers_d.get(),
            bits.get(),
            i,
            prefixMemory[0].get(),
            numbers_d.count
        );

        cudaKernelChecker.check("initBitScan");

        Scan();
    
        placeNumbersByScan<<<GRID_SIZE, BLOCK_SIZE>>>(
            numbers_d.get(),
            newNumbers_d.get(),
            bits.get(),
            prefixMemory[0].get(),
            numbers_d.count
        );

        cudaKernelChecker.check("placeNumbersByScan");

        numbers_d.Swap(numbers_d, newNumbers_d);
    }

    numbers_d.memcpy(numbers.data(), cudaMemcpyDeviceToHost);
}

void Lab05::PrintTextResult()
{
    /* std::vector<uint> data_h(data_d.count); */

    /* data_d.memcpy(data_h.data(), cudaMemcpyDeviceToHost); */
    
    timer.stop();

    /* print_vector1d(numbers); */

    timer.print_time();
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
