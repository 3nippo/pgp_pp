#pragma once

#include "Material.cuh.cu"
#include "HitRecord.cuh.cu"
#include "Ray.cuh.cu"
#include "utils.cuh.cu"

#include <curand.h>
#include <curand_kernel.h>

namespace RayTracing
{

class Lambertian : public Material
{
protected:
    const Texture* const * const m_albedo;
    curandState *m_states;
    
public:
    __host__ __device__
    Lambertian(
        const float transparency,
        const float reflectance,
        const Texture* const * const albedo,
        curandState *states
    )
        : Material(transparency, reflectance),
          m_albedo(albedo),
          m_states(states)
    {}

private:
    __host__ __device__
    Vector3 RandomUnitSphereSurfaceVector() const
    {
        #ifdef __CUDA_ARCH__
        
        int id = threadIdx.x + blockDim.x * blockIdx.x;

        while (true)
        {
            Vector3 v{
                curand_uniform(m_states + id) * 2 - 1,
                curand_uniform(m_states + id) * 2 - 1,
                curand_uniform(m_states + id) * 2 - 1
            };

            if (v.LengthSquared() > 1)
                continue;

            return v.UnitVector();
        }

        #else

        while (true)
        {
            Vector3 v{
                GenRandom(-1, 1),
                GenRandom(-1, 1),
                GenRandom(-1, 1),
            };

            if (v.LengthSquared() > 1)
                continue;

            return v.UnitVector();
        }

        #endif
    }
    
    __host__ __device__
    virtual bool Scatter(
        const Ray &ray,
        const HitRecord &hitRecord,
        Color &attenuation,
        Ray &scattered
    ) const override
    {
        Vector3 scatterDir = hitRecord.normal + RandomUnitSphereSurfaceVector();

        if (scatterDir.NearZero())
            scatterDir = hitRecord.normal;

        scattered = Ray(
            hitRecord.point,
            scatterDir
        );

        attenuation = (*m_albedo)->GetColor(hitRecord.u, hitRecord.v);

        return true;
    }
};

} // namespace RayTracing
