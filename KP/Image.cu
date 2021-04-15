#include "Image.cuh.cu"

#include <fstream>

#include "utils.cuh.cu"
#include "./dummy_helper.cuh.cu"

namespace RayTracing
{

Image::Image(const std::string &fileName)
{
    std::ifstream file(fileName);
    
    file.read(reinterpret_cast<char*>(&m_width), sizeof(m_width));
    file.read(reinterpret_cast<char*>(&m_height), sizeof(m_height));

    for (int i = 0; i < m_width * m_height; ++i)
    {
        unsigned char r = 0,
                      g = 0,
                      b = 0,
                      a = 0;

        file.read(reinterpret_cast<char*>(&r), sizeof(r));
        file.read(reinterpret_cast<char*>(&g), sizeof(g));
        file.read(reinterpret_cast<char*>(&b), sizeof(b));
        file.read(reinterpret_cast<char*>(&a), sizeof(a));

        buffer.push_back(Color{
            r / 255.0f,
            g / 255.0f,
            b / 255.0f
        });
    }
}

template<>
void Image::Init<true>()
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    checkCudaErrors(cudaMallocArray(
        &m_buffer_d, 
        &channelDesc, 
        m_width,
        m_height
    ));

    checkCudaErrors(cudaMemcpyToArray(
        m_buffer_d,
        0,
        0,
        buffer.data(),
        buffer.size() * sizeof(float4),
        cudaMemcpyHostToDevice
    ));
    
    memset(&m_cudaTextureResourceDesc, 0, sizeof(m_cudaTextureResourceDesc));

    m_cudaTextureResourceDesc.resType            = cudaResourceTypeArray;
    m_cudaTextureResourceDesc.res.array.array    = m_buffer_d;

    memset(&m_cudaTextureDesc, 0, sizeof(m_cudaTextureDesc));

    m_cudaTextureDesc.normalizedCoords = true;
    m_cudaTextureDesc.filterMode       = cudaFilterModePoint;
    m_cudaTextureDesc.addressMode[0] = cudaAddressModeClamp;
    m_cudaTextureDesc.addressMode[1] = cudaAddressModeClamp;
    m_cudaTextureDesc.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(
        &cudaTexture,
        &m_cudaTextureResourceDesc,
        &m_cudaTextureDesc,
        NULL
    ));
}

template<>
cudaTextureObject_t Image::GetResource<true>()
{
    return cudaTexture;
}

template<>
Image Image::GetResource<false>()
{
    return *this;
}

template<>
void Image::Init<false>() {}

Color Image::GetColor(const float u, const float v) const
{
    int w = Clamp(u * (m_width - 1), 0, (m_width - 1)),
        h = Clamp(v * (m_height - 1), 0, (m_height - 1));

    return buffer[w + m_width * h];
}

void Image::Deinit()
{
    if (m_buffer_d != nullptr)
    {
        checkCudaErrors(cudaFreeArray(m_buffer_d));
        checkCudaErrors(cudaDestroyTextureObject(cudaTexture));
    }
}


} // RayTracing
