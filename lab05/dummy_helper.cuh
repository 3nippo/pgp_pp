#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>


template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) 
    {
        throw std::runtime_error(
            std::string("ERROR: ") 
            + file 
            + ":" 
            + std::to_string(line)
            + "\n"
            + cudaGetErrorName(err) 
            + " " 
            + "\n"
            + cudaGetErrorString(err) 
            + " " 
            + func
        );
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "ERROR: %s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// This will only print the proper error string when calling cudaGetLastError
// but not exit program incase error detected.
#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)

inline void __printLastCudaError(const char *errorMessage, const char *file,
                                 const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "ERROR: %s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
  }
}

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif


template <class T>
std::vector<T> read_vector1d(size_t n)
{
    std::vector<T> v;

    for (size_t i = 0; i < n; ++i)
    {
        T input;
        std::cin >> input;

        v.push_back(input);
    }

    return v;
}


template <class T>
void print_vector1d(std::vector<T> &v)
{
    for (size_t i = 0; i + 1 < v.size(); ++i)
        std::cout << v[i] << ' ';
    
    if (!v.empty())
        std::cout << v.back() << std::endl;
}


template <class T>
class CudaMemory
{
private:
    T *ptr = nullptr;

public:
    size_t count = 0;

    static void Swap(CudaMemory &left, CudaMemory &right)
    {
        size_t saveCount = left.count;
        left.count = right.count;
        right.count = saveCount;

        T *savePtr = left.ptr;
        left.ptr = right.ptr;
        right.ptr = savePtr;
    }
    
    CudaMemory(const CudaMemory &right)
    {
        ptr = right.ptr;
        count = right.count;
    }

    CudaMemory& operator=(const CudaMemory &right)
    {
        ptr = right.ptr;
        count = right.count;
    }
    
    void Reset()
    {
        ptr = nullptr;
        count = 0;
    }

    CudaMemory() {}

    void alloc(size_t count)
    {
        if (count == 0)
            return;

        this->count = count;

        checkCudaErrors(
            cudaMalloc(&ptr, count * sizeof(T))
        );
    }

    CudaMemory(size_t count)
    {
        alloc(count);   
    }

    T*& get()
    {
        return ptr;
    }

    size_t bytes_size()
    {
        return count * sizeof(T);
    }

    void memcpy(void *ptr, cudaMemcpyKind kind, size_t deviceOffset=0, size_t count=0)
    {
        if (!ptr)
            return;

        void *dst, *src;

        if (kind == cudaMemcpyHostToDevice)
        {
            src = ptr;
            dst = reinterpret_cast<T*>(this->ptr) + deviceOffset;
        }
        else
        {
            src = reinterpret_cast<T*>(this->ptr) + deviceOffset;
            dst = ptr;
        }

        size_t count_to_cpy = (count ? count : this->count);

        checkCudaErrors(
            cudaMemcpy(dst, src, count_to_cpy * sizeof(T), kind)
        );
    }

    void dealloc()
    {
        if (!ptr)
            return;

        checkCudaErrors(
            cudaFree(ptr)
        );
    }

    ~CudaMemory()
    {
        dealloc();
    }
};


class CudaKernelChecker
{
private:
    cudaError_t err;

public:
    CudaKernelChecker()
    {
        err = cudaSuccess;
    }

    void check(const char *name)
    {
        err = cudaGetLastError();
        
        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("ERROR: Failed to launch kernel ")
                + name
                + " (error "
                + cudaGetErrorString(err)
                + ")!"    
            );
        }
    }
};


class CudaTimer
{
private:
    cudaEvent_t event_start, event_stop;
public:
    CudaTimer()
    {
        checkCudaErrors(cudaEventCreate(&event_start));
        checkCudaErrors(cudaEventCreate(&event_stop));
    }

    void start()
    {
        checkCudaErrors(cudaEventRecord(event_start));
    }

    void stop()
    {
        checkCudaErrors(cudaEventRecord(event_stop));
        checkCudaErrors(cudaEventSynchronize(event_stop));
    }
    
    float get_time()
    {
        float ms;

        checkCudaErrors(cudaEventElapsedTime(&ms, event_start, event_stop));

        return ms;
    }

    void print_time()
    {
        float ms_time = get_time();

        std::cout << "time: " << ms_time << std::endl;
    }
};
