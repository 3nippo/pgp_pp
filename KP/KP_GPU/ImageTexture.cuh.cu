#pragma once

#include <string>

#include "Image.cuh.cu"
#include "Texture.cuh.cu"
#include "Vector3.cuh.cu"

namespace RayTracing
{

class ImageTexture : public Texture
{
private:
    const cudaTextureObject_t m_cudaTexture;
    const Color m_color;

public:
    __host__ __device__
    ImageTexture(
        const cudaTextureObject_t cudaTexture,
        const Color &color
    )
        : m_cudaTexture(cudaTexture),
          m_color(color)
    {}

private:
    __device__
    virtual Color GetColor(const float u, const float v) const override
    {
        return m_color * tex2D<float4>(m_cudaTexture, u, v);
    }
};

};
