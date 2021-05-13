#include "RayTracer.cuh.cu"

#include <fstream>
#include <iostream>
#include <string>

#include "HitRecord.cuh.cu"
#include "utils.cuh.cu"

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

namespace RayTracing
{

template<typename T>
__global__
void SetValues(
    T *data,
    T value,
    const size_t count
)
{
    const size_t offset = gridDim.x * blockDim.x;

    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count; i += offset)
        data[i] = value;
}

__global__
void ComputeNextRays(
    const PolygonsManager<true> polygonsManager,
    const RayTraceData* const raysData,
    const int* const raysDataKeys,
    const int raysCount,
    RayTraceData* const newRaysData,
    int* const newRaysDataKeys,
    float4* const picture,
    const int width
)
{
    const int offset = gridDim.x * blockDim.x;

    for (int id = threadIdx.x + blockIdx.x * blockDim.x; id < raysCount; id += offset)
    {
        HitRecord hitRecord;

        hitRecord.t = INF;
        
        const Ray &ray = raysData[id].scattered;

        if (!polygonsManager.Hit(ray, 0.001, hitRecord))
        {
            /* float4 *pixel = picture + raysDataKeys[id] + raysData[id].h * width; */

            /* Color emittedColor = Vector3(1, 1, 1) * raysData[id].attenuation; */

            /* atomicAdd(&pixel->x, emittedColor.d.x); */
            /* atomicAdd(&pixel->y, emittedColor.d.y); */
            /* atomicAdd(&pixel->z, emittedColor.d.z); */

            continue;
        }
        
        if (hitRecord.material->Emits())
        {
            float4 *pixel = picture + raysDataKeys[id] + raysData[id].h * width;
            
            Color emittedColor = hitRecord.material->Emitted(hitRecord) * raysData[id].attenuation;

            atomicAdd(&pixel->x, emittedColor.d.x);
            atomicAdd(&pixel->y, emittedColor.d.y);
            atomicAdd(&pixel->z, emittedColor.d.z);
        }

        if (
            hitRecord.material->reflectance != 0
            && hitRecord.material->Scatter(
                ray,
                hitRecord,
                newRaysData[2*id].attenuation,
                newRaysData[2*id].scattered
            )
        )
        {
            newRaysData[2*id].attenuation *= raysData[id].attenuation * hitRecord.material->reflectance;
            
            if (!newRaysData[2*id].attenuation.NearZero())
            {
                newRaysDataKeys[2*id] = raysDataKeys[id];
                newRaysData[2*id].h = raysData[id].h;
            }
        }

        newRaysData[2*id+1].attenuation = raysData[id].attenuation * hitRecord.material->transparency;

        if (
            hitRecord.material->transparency != 0
            && !newRaysData[2*id+1].attenuation.NearZero()
        )
        {
            newRaysDataKeys[2*id+1] = raysDataKeys[id];
            newRaysData[2*id+1].h = raysData[id].h;
            newRaysData[2*id+1].scattered = Ray(hitRecord.point, ray.direction);
        }
    }
}

RayTracer::RayTracer(
    const Config &config,
    const float start, 
    const float end 
)
    : m_config(config),
      m_camera(
          m_config.width,
          m_config.height,
          m_config.horizontalViewDegrees
      ),
      m_normalizer(config.samplesPerPixel),
      m_start(start),
      m_end(end)
{}

void RayTracer::FillRaysData(
    std::vector<RayTraceData> &raysData, 
    std::vector<int> &raysDataKeys
)
{
    int height = m_config.height;
    int width = m_config.width;

    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            for (int s = 0; s < m_config.samplesPerPixel; ++s)
            {
                float y = (height - 1 - h + GenRandom()) / (height - 1),
                      x = (w + GenRandom()) / (width - 1);

                Ray ray = m_camera.GetRay(x, y);
                
                raysDataKeys.push_back(w);
                raysData.push_back(RayTraceData{
                    h,
                    {1, 1, 1},
                    ray
                });
            }
        }
    }
}


template<>
size_t RayTracer::Render<true>(
    const PolygonsManager<true> &polygonsManager,
    std::vector<RayTraceData> &raysData,
    std::vector<int> &raysDataKeys,
    std::vector<float4> &picture
)
{
    const int lastSize = raysData.size() % CAP,
              blocksNum = raysData.size() / CAP + (lastSize != 0);
        
    CudaMemory<float4> picture_d(picture.size());

    picture_d.memcpy(picture.data(), cudaMemcpyHostToDevice);

    size_t frameRaysNum = 0;

    for (int i = 0; i < blocksNum; ++i)
    {
        std::cerr << ">>> block " << i+1 << "/" << blocksNum << std::endl << std::endl;  

        int depth = m_config.recursionDepth;
        int count = (i == blocksNum - 1 && lastSize != 0 ? lastSize : CAP);
        
        CudaMemory<RayTraceData> currentRaysData_d(count);
        CudaMemory<int> currentRaysDataKeys_d(count);

        currentRaysData_d.memcpy(raysData.data() + i * CAP, cudaMemcpyHostToDevice);
        currentRaysDataKeys_d.memcpy(raysDataKeys.data() + i * CAP, cudaMemcpyHostToDevice);

        CudaKernelChecker checker;
        std::string kernelName = "ComputeNextRays, depth: ";

        CudaMemory<RayTraceData> newRaysData_d;
        CudaMemory<int> newRaysDataKeys_d;

        while (depth--)
        {
            std::cerr << "> depth " << m_config.recursionDepth - depth << "/" << m_config.recursionDepth << std::endl;
            std::cerr << "> rays count " << count << std::endl;

            if (count == 0)
            {
                std::cerr << "> rays count is 0, end work for this block" << std::endl << std::endl;
                break;
            }

            if (count > 5 * CAP)
            {
                std::cerr << "> rays count is too big, end work for this block" << std::endl << std::endl;
                break;
            }

            if (depth == 0)
                std::cerr << std::endl;

            if (count > newRaysData_d.count / 2)
            {
                newRaysData_d.dealloc();
                newRaysData_d.alloc(count * 2);

                newRaysDataKeys_d.dealloc();
                newRaysDataKeys_d.alloc(count * 2);
            }

            SetValues<<<GRID_SIZE, BLOCK_SIZE>>>(
                newRaysDataKeys_d.get(),
                INT_INF,
                newRaysDataKeys_d.count
            );

            // render step
            ComputeNextRays<<<GRID_SIZE, BLOCK_SIZE>>>(
                polygonsManager,
                currentRaysData_d.get(),
                currentRaysDataKeys_d.get(),
                count,
                newRaysData_d.get(),
                newRaysDataKeys_d.get(),
                picture_d.get(),
                m_config.width
            );

            frameRaysNum += count;

            std::string fullKernelName = kernelName + std::to_string(depth + 1);
        
            checker.check(fullKernelName);

            // compact
            thrust::device_ptr<RayTraceData> newRaysData = thrust::device_pointer_cast(newRaysData_d.get());
            thrust::device_ptr<int> newRaysDataKeys = thrust::device_pointer_cast(newRaysDataKeys_d.get());

            thrust::stable_sort_by_key(newRaysDataKeys, newRaysDataKeys + newRaysDataKeys_d.count, newRaysData);

            // get size
            thrust::device_ptr<int> infStart = thrust::lower_bound(
                newRaysDataKeys, 
                newRaysDataKeys + newRaysDataKeys_d.count,
                INT_INF,
                thrust::less<int>()
            );
            
            // update count
            count = infStart - newRaysDataKeys;
            
            CudaMemory<RayTraceData>::Swap(currentRaysData_d, newRaysData_d);
            CudaMemory<int>::Swap(currentRaysDataKeys_d, newRaysDataKeys_d);
        }
    }

    picture_d.memcpy(picture.data(), cudaMemcpyDeviceToHost);

    return frameRaysNum;
}

template<>
size_t RayTracer::Render<false>(
    const PolygonsManager<false> &polygonsManager,
    std::vector<RayTraceData> &raysData,
    std::vector<int> &raysDataKeys,
    std::vector<float4> &picture
)
{
    std::vector<RayTraceData> newRaysData;
    std::vector<int> newRaysDataKeys;
    
    size_t frameRaysNum = 0;

    for (int depth = 0; depth < m_config.recursionDepth; ++depth)
    {
        std::cerr << "> depth " << depth + 1 << "/" << m_config.recursionDepth << std::endl << std::endl;

        for (int i = 0; i < raysData.size(); ++i)
        {
            HitRecord hitRecord;

            hitRecord.t = INF;
            
            const Ray &ray = raysData[i].scattered;

            if (!polygonsManager.Hit(ray, 0.001, hitRecord))
            {
                /* float4 *pixel = picture + raysDataKeys[id] + raysData[id].h * width; */

                /* Color emittedColor = Vector3(1, 1, 1) * raysData[id].attenuation; */

                /* atomicAdd(&pixel->x, emittedColor.d.x); */
                /* atomicAdd(&pixel->y, emittedColor.d.y); */
                /* atomicAdd(&pixel->z, emittedColor.d.z); */

                continue;
            }
            
            if (hitRecord.material->Emits())
            {
                float4 &pixel = picture[raysDataKeys[i] + raysData[i].h * m_config.width];
                
                Color emittedColor = hitRecord.material->Emitted(hitRecord) * raysData[i].attenuation;
                
                pixel.x += emittedColor.d.x;
                pixel.y += emittedColor.d.y;
                pixel.z += emittedColor.d.z;
            }

            newRaysData.emplace_back();

            if (
                hitRecord.material->reflectance != 0
                && hitRecord.material->Scatter(
                    ray,
                    hitRecord,
                    newRaysData.back().attenuation,
                    newRaysData.back().scattered
                )
            )
            {
                newRaysData.back().attenuation *= raysData[i].attenuation * hitRecord.material->reflectance;
                
                if (!newRaysData.back().attenuation.NearZero())
                {
                    newRaysDataKeys.push_back(raysDataKeys[i]);
                    newRaysData.back().h = raysData[i].h;
                }
            }
            else
                newRaysData.pop_back();
            
            newRaysData.emplace_back();
            newRaysData.back().attenuation = raysData[i].attenuation * hitRecord.material->transparency;

            if (
                hitRecord.material->transparency != 0
                && !newRaysData.back().attenuation.NearZero()
            )
            {
                newRaysDataKeys.push_back(raysDataKeys[i]);
                newRaysData.back().h = raysData[i].h;
                newRaysData.back().scattered = Ray(hitRecord.point, ray.direction);
            }
            else
                newRaysData.pop_back();
        }

        frameRaysNum += raysData.size();

        std::swap(newRaysData, raysData);
        std::swap(newRaysDataKeys, raysDataKeys);

        newRaysData.clear();
        newRaysDataKeys.clear();
    }

    return frameRaysNum;
}

void RayTracer::SetupCamera(const float t)
{
    auto TrajectoryToPoint = [] (const float t, const Trajectory &trajectory) -> Point3
    {
        return Point3{
            trajectory.r + trajectory.rA * sinf(trajectory.rOm * t + trajectory.rP),
            trajectory.z + trajectory.zA * sinf(trajectory.zOm * t + trajectory.zP),
            trajectory.phi + trajectory.phiOm * t
        };
    };

    auto ToCartesian = [] (const Point3 &point) -> Point3
    {
        return Point3{
            point.d.x * cosf(point.d.z),
            point.d.y,
            point.d.x * sinf(point.d.z)
        };
    };

    Vector3 lookAt = ToCartesian(TrajectoryToPoint(t, m_config.lookAt)),
            lookFrom = ToCartesian(TrajectoryToPoint(t, m_config.lookFrom));

    m_camera.LookAt(
        lookAt,
        lookFrom
    );
}

std::string RayTracer::GetFrameName(const int index)
{
    char nameBuffer[CONFIG_STRING_MAX_COUNT];

    sprintf(nameBuffer, m_config.outputTemplate, index);

    return nameBuffer;
}

void RayTracer::WriteToFile(const std::string &frameName, const std::vector<float4> &picture)
{
    std::ofstream outputFile(frameName);
    
    int height = m_config.height;
    int width = m_config.width;

    outputFile.write(
        reinterpret_cast<const char*>(&width), 
        sizeof(width)
    );

    outputFile.write(
        reinterpret_cast<const char*>(&height),
        sizeof(height)
    );

    for (size_t i = 0; i < picture.size(); ++i)
        outputFile << picture[i];
}

void RayTracer::WriteToFilePPM(const std::string &frameName, const std::vector<float4> &picture)
{
    std::ofstream outputFile(frameName);

    outputFile << "P3" << std::endl;

    outputFile << m_config.width << ' ' << m_config.height << std::endl;
    
    const int maxChannelValue = 255;

    outputFile << maxChannelValue << std::endl;

    for (size_t i = 0; i < picture.size(); ++i)
    {
        unsigned char r, g, b, a;

        ColorToRGBA(picture[i], r, g, b, a);

        outputFile << (int)r << ' ' << (int)g << ' ' << (int)b << std::endl;
    }
}

} // namespace RayTracing
