#pragma once

#include <vector>
#include <string>

#include "Vector3.cuh.cu"

namespace RayTracing
{
class Image
{
private:
    int m_width;
    int m_height;

    cudaResourceDesc m_cudaTextureResourceDesc;
    cudaTextureDesc m_cudaTextureDesc;
    cudaArray *m_buffer_d;

public:
    std::vector<Color> buffer;
    cudaTextureObject_t cudaTexture;

public:
    Image(const std::string &fileName);

    ~Image();
};
};
