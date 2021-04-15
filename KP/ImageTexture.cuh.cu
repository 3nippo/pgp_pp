#pragma once

#include <string>

#include "Image.cuh.cu"
#include "Texture.cuh.cu"
#include "Vector3.cuh.cu"

namespace RayTracing
{

template<bool isGPU>
class ImageTexture
{

};

template<>
class ImageTexture<true> : public Texture
{
private:
    const cudaTextureObject_t m_texResource;
    const Color m_color;

public:
    __host__ __device__
    ImageTexture(
        const cudaTextureObject_t cudaTexture,
        const Color &color
    )
        : m_texResource(cudaTexture),
          m_color(color)
    {}

private:
    __host__ __device__
    virtual Color GetColor(const float u, const float v) const override
    {
        #ifdef __CUDA_ARCH__
        return m_color * tex2D<float4>(m_texResource, u, v);
        #else
        return Color();
        #endif
    }
};

template<>
class ImageTexture<false> : public Texture
{
private:
    const Image m_texResource;
    const Color m_color;

public:
    ImageTexture(
        const Image image,
        const Color &color
    )
        : m_texResource(image),
          m_color(color)
    {}

private:
    __host__ __device__
    virtual Color GetColor(const float u, const float v) const override
    {
        #ifdef __CUDA_ARCH__
        return Color();
        #else
        return m_color * m_texResource.GetColor(u, v);
        #endif
    }
};

};
