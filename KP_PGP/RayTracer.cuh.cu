#pragma once

#include <vector>
#include <string>
#include <cassert>
#include <algorithm>
#include <iostream>

#include "Config.cuh.cu"
#include "Vector3.cuh.cu"
#include "Ray.cuh.cu"
#include "PolygonsManager.cuh.cu"
#include "dummy_helper.cuh.cu"
#include "Camera.cuh.cu"

namespace RayTracing
{

constexpr int CAP = 1000000;
constexpr int GRID_SIZE = 128;
constexpr int BLOCK_SIZE = 512; 

struct Normalizer
{
    int samplesPerPixel;

    Normalizer(const int samplesPerPixel) : samplesPerPixel(samplesPerPixel) {}
    
    float4 operator()(const float4 &v) const
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
    const Config &m_config;
    Camera m_camera;
    const float m_start, m_end;

    const Normalizer m_normalizer;

public:
    RayTracer(
        const Config &config,
        const float start,
        const float end
    );
    
    template<bool isGPU>
    void RenderFrames(const PolygonsManager<isGPU> &polygonsManager)
    {
        std::vector<float4> picture;
        
        float step = 2 * M_PI / m_config.framesNum;

        for (int i = m_start; i < m_end; ++i)
        {
            picture.assign(m_config.width * m_config.height, { 0, 0, 0, 0 });

            const float t = i * step;

            SetupCamera(t);

            std::vector<RayTraceData> raysData;
            std::vector<int> raysDataKeys;

            FillRaysData(raysData, raysDataKeys);

            CudaTimer cudaTimer;
            cudaTimer.start();

            size_t numberOfRays = Render<isGPU>(
                polygonsManager, 
                raysData, 
                raysDataKeys, 
                picture
            );

            cudaTimer.stop();

            float frameRenderTime = cudaTimer.get_time();
            
            printf("**********************************************\n");
            printf("\n\nFrame %d\tTime: %fms\tNumber of rays: %zu\n\n", i+1, frameRenderTime, numberOfRays);
            printf("**********************************************\n");

            std::for_each(
                picture.begin(),
                picture.end(),
                [this] (float4 &v)
                {
                    v = (Vector3(v) / m_config.samplesPerPixel).Clamp(0, 1).d; 
                }
            );

            WriteToFilePPM(GetFrameName(i), picture);
        }
    }

private:
    void SetupCamera(const float t);
    void FillRaysData(std::vector<RayTraceData> &raysData, std::vector<int> &raysDataKeys);
    void WriteToFile(const std::string &frameName, const std::vector<float4> &picture);
    void WriteToFilePPM(const std::string &frameName, const std::vector<float4> &picture);
    std::string GetFrameName(const int index);
    
    template<bool isGPU>
    size_t Render(
        const PolygonsManager<isGPU> &polygonsManager,
        std::vector<RayTraceData> &raysData,
        std::vector<int> &raysDataKeys,
        std::vector<float4> &picture
    )
    {
        assert(("Not implemented", false));
    }
};

} // namespace RayTracing
