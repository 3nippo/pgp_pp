#pragma once

#include <vector>
#include <string>
#include <cassert>

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
    cudaArray *m_buffer_d = nullptr;
public:
    std::vector<Color> buffer;
    cudaTextureObject_t cudaTexture;

public:
    Image(const std::string &fileName);

    template<bool isGPU>
    void Init()
    {
        assert(("Not implemented", false));
    }

    template<bool isGPU, typename T>
    T GetResource()
    {
        assert(("Not implemented", false));
    }


    Color GetColor(const float u, const float v) const;

    void Deinit();
};
};
