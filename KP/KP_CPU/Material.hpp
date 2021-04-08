#pragma once

#include "Ray.hpp"
#include "HitRecord.hpp"
#include "Vector3.hpp"
#include "Texture.hpp"

namespace RayTracing
{

class Material
{
public:
    float transparency;
    float reflectance;
public:
    Material(
        const float transparency,
        const float reflectance
    )
        : transparency(transparency),
          reflectance(reflectance)
    {}

    virtual bool Scatter(
        const Ray &ray,
        const HitRecord &hitRecord,
        Color &attenuation,
        Ray &scattered
    ) const = 0;

    virtual Color Emitted(
        const HitRecord &hitRecord
    ) const
    {
        return Color(0, 0, 0);
    }
};

} // namespace RayTracing
