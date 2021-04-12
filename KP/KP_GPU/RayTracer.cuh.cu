#pragma once

#include <vector>
#include <string>

#include "Camera.cuh.cu"
#include "Vector3.cuh.cu"
#include "Scene.cuh.cu"
#include "Ray.cuh.cu"
#include "./dummy_helper.cuh.cu"

namespace RayTracing
{
constexpr int CAP = 1000000;
constexpr int GRID_SIZE = 128;
constexpr int BLOCK_SIZE = 512; 


struct Normalizer
{
    int samplesPerPixel;

    Normalizer(const int samplesPerPixel) : samplesPerPixel(samplesPerPixel) {}
    
    __device__
    float4 operator()(const float4 &v)
    {
        return (Vector3(v) / samplesPerPixel).Clamp(0, 1).d; 
    }
};

struct RayTraceData
{
    int h;
    Vector3 attenuation;
    Ray scattered;
};

class RayTracer
{
private:
    const int m_width;
    const int m_height;
    const int m_samplesPerPixel;
    const int m_depth;

    const Camera &m_camera;
    const Scene &m_scene;
    std::vector<float4> m_pictureBuffer;
    std::vector<RayTraceData> m_raysData;
    std::vector<int> m_raysDataKeys;


    CudaMemory<float4> m_pictureBuffer_d;

public:
    RayTracer(
        const Camera &camera,
        const Scene &scene,
        const int width,
        const int height,
        const int samplesPerPixel,
        const int depth
    );

    void Render();

    void WriteToFile(const std::string &name);
    void WriteToFilePPM(const std::string &name);

private:
    Color RayColor(const Ray &ray, const int depth);
    void CopyPictureToHost();
};

} // namespace RayTracing
