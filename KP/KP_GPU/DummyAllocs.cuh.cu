#pragma once

#include "./dummy_helper.cuh.cu"

namespace RayTracing
{

template<typename Derived, typename Base, typename ... Args>
__global__
void __AllocInstance(
    Base** _texture,
    Args ... args
)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *_texture = new Derived(args...);
}

template<typename Base>
__global__
void __DeallocInstance(Base** _texture)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        delete *_texture;
}

template<typename Derived, typename Base, typename ... Args>
void AllocInstance(
    Base** _texture,
    Args ... args
)
{
    CudaKernelChecker checker;

    __AllocInstance<Derived, Base, Args...><<<1, 32>>>(_texture, args...);

    checker.check("AllocInstance");
}

template<typename Base>
void DeallocInstance(Base** _texture)
{
    CudaKernelChecker checker;

    __DeallocInstance<<<1, 32>>>(_texture);

    checker.check("DeallocInstance");
}

template<typename Derived, typename Base, typename ... Args>
class CudaHeapObject
{
public:
    Base **ptr = nullptr;
public:
    CudaHeapObject(Args ... args)
    {
        checkCudaErrors(cudaMalloc(&ptr, sizeof(Base**)));

        AllocInstance<Derived>(ptr, args...);
    }
    
    ~CudaHeapObject()
    {
        DeallocInstance(ptr);

        checkCudaErrors(cudaFree(ptr));
    }
};
} // namespace RayTracing
