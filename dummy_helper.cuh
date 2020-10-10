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
    for (size_t i = 0; i < v.size() - 1; ++i)
        std::cout << v[i] << ' ';
    std::cout << v.back() << std::endl;
}


template <class T>
class CudaMemory
{
private:
    T *ptr;

public:
    size_t count;

    CudaMemory(size_t count) : count(count)
    {
        checkCudaErrors(
            cudaMalloc(&ptr, count * sizeof(T))
        );    
    }

    T*& get()
    {
        return ptr;
    }

    size_t bytes_size()
    {
        return count * sizeof(T);
    }

    void memcpy(void *ptr, cudaMemcpyKind kind, size_t count=0)
    {
        void *dst, *src;

        if (kind == cudaMemcpyHostToDevice)
        {
            src = ptr;
            dst = this->ptr;
        }
        else
        {
            src = this->ptr;
            dst = ptr;
        }

        size_t count_to_cpy = (count ? count : this->count);

        checkCudaErrors(
            cudaMemcpy(dst, src, count_to_cpy * sizeof(T), kind)
        );
    }

    ~CudaMemory()
    {
        checkCudaErrors(
            cudaFree(ptr)
        );
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
