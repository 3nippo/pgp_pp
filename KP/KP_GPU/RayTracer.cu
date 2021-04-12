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
#include <thrust/transform.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>

namespace RayTracing
{

__global__
void RandomShuffle(
    RayTraceData *raysData,
    int *raysDataKeys,
    const int raysCount
)
{
    if (threadIdx.x != 0 || blockDim.x != 0)
        return;

    thrust::minstd_rand rng;
    
    thrust::uniform_int_distribution<int> dist(0, raysCount-1);
    
    for (int i = 0; i < raysCount; ++i)
    {
        int generated = dist(rng);

        {
            RayTraceData tmp = raysData[i];
            raysData[i] = raysData[generated];
            raysData[generated] = tmp;
        }

        {
            int tmp = raysDataKeys[i];
            raysDataKeys[i] = raysDataKeys[generated];
            raysDataKeys[generated] = tmp;
        }
    }
}

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
    Scene scene,
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

        if (!scene.Hit(ray, 0.001, hitRecord))
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
    const Camera &camera,
    const Scene &scene,
    const int width,
    const int height,
    const int samplesPerPixel,
    const int depth
)
    : m_camera(camera), 
      m_scene(scene),
      m_width(width), 
      m_height(height), 
      m_samplesPerPixel(samplesPerPixel),
      m_depth(depth)
{
    for (int h = 0; h < m_height; ++h)
    {
        for (int w = 0; w < m_width; ++w)
        {
            for (int s = 0; s < m_samplesPerPixel; ++s)
            {
                float y = (m_height - 1 - h + GenRandom()) / (m_height - 1),
                      x = (w + GenRandom()) / (m_width - 1);

                Ray ray = m_camera.GetRay(x, y);
                
                m_raysDataKeys.push_back(w);
                m_raysData.push_back(RayTraceData{
                    h,
                    {1, 1, 1},
                    ray
                });
            }
        }
    }
    
    m_pictureBuffer.resize(m_height * m_width);
    m_pictureBuffer_d.alloc(m_height * m_width);
    SetValues<<<GRID_SIZE, BLOCK_SIZE>>>(
        m_pictureBuffer_d.get(), 
        { 0, 0, 0, 0}, 
        m_pictureBuffer_d.count
    );
}


void RayTracer::Render()
{
    const int lastSize = m_raysData.size() % CAP,
              blocksNum = m_raysData.size() / CAP + (lastSize != 0);

    for (int i = 0; i < blocksNum; ++i)
    {
        std::cerr << ">>> block " << i+1 << "/" << blocksNum << std::endl << std::endl;  

        int depth = m_depth;
        int count = (i == blocksNum - 1 && lastSize != 0 ? lastSize : CAP);
        
        CudaMemory<RayTraceData> currentRaysData_d(count);
        CudaMemory<int> currentRaysDataKeys_d(count);

        currentRaysData_d.memcpy(m_raysData.data() + i * CAP, cudaMemcpyHostToDevice);
        currentRaysDataKeys_d.memcpy(m_raysDataKeys.data() + i * CAP, cudaMemcpyHostToDevice);

        CudaKernelChecker checker;
        std::string kernelName = "ComputeNextRays, depth: ";

        CudaMemory<RayTraceData> newRaysData_d;
        CudaMemory<int> newRaysDataKeys_d;

        while (depth--)
        {
            std::cerr << "> depth " << m_depth - depth << "/" << m_depth << std::endl;
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
                /* if (count > CAP) */
                /* { */
                /*     RandomShuffle<<<GRID_SIZE, BLOCK_SIZE>>>( */
                /*         currentRaysData_d.get(), */
                /*         currentRaysDataKeys_d.get(), */
                /*         count */
                /*     ); */

                /*     count = CAP; */
                /* } */

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
                m_scene,
                currentRaysData_d.get(),
                currentRaysDataKeys_d.get(),
                count,
                newRaysData_d.get(),
                newRaysDataKeys_d.get(),
                m_pictureBuffer_d.get(),
                m_width
            );

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
}

void RayTracer::CopyPictureToHost()
{
    thrust::device_ptr<float4> picture_d = thrust::device_pointer_cast(m_pictureBuffer_d.get());
    
    Normalizer normalizer(m_samplesPerPixel);

    thrust::transform(
        picture_d,
        picture_d + m_pictureBuffer_d.count,
        picture_d,
        normalizer
    );

    m_pictureBuffer_d.memcpy(m_pictureBuffer.data(), cudaMemcpyDeviceToHost);
}

void RayTracer::WriteToFile(const std::string &name)
{
    CopyPictureToHost();

    std::ofstream outputFile(name);

    outputFile.write(
        reinterpret_cast<const char*>(&m_width), 
        sizeof(m_width)
    );

    outputFile.write(
        reinterpret_cast<const char*>(&m_height),
        sizeof(m_height)
    );

    for (size_t i = 0; i < m_pictureBuffer.size(); ++i)
        outputFile << m_pictureBuffer[i];
}

void RayTracer::WriteToFilePPM(const std::string &name)
{
    CopyPictureToHost();

    std::ofstream outputFile(name);

    outputFile << "P3" << std::endl;

    outputFile << m_width << ' ' << m_height << std::endl;

    outputFile << 255 << std::endl;

    for (size_t i = 0; i < m_pictureBuffer.size(); ++i)
    {
        unsigned char r, g, b, a;

        ColorToRGBA(m_pictureBuffer[i], r, g, b, a);

        outputFile << (int)r << ' ' << (int)g << ' ' << (int)b << std::endl;
    }
}

} // namespace RayTracing
