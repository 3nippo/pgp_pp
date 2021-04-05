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
