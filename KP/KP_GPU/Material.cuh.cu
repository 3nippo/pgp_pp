#pragma once

#include "Ray.cuh.cu"
#include "HitRecord.cuh.cu"
#include "Vector3.cuh.cu"
#include "Texture.cuh.cu"

namespace RayTracing
{

class Material
{
public:
    float transparency;
    float reflectance;
public:
    __host__ __device__
    Material(
        const float transparency,
        const float reflectance
    )
        : transparency(transparency),
          reflectance(reflectance)
    {}
    
    __device__
    virtual bool Scatter(
        const Ray &ray,
        const HitRecord &hitRecord,
        Color &attenuation,
        Ray &scattered
    ) const = 0;

    __device__
    virtual Color Emitted(
        const HitRecord &hitRecord
    ) const
    {
        return Color(0, 0, 0);
    }

    __device__
    virtual bool Emits() const
    {
        return false;
    }
};

} // namespace RayTracing
