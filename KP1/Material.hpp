#pragma once

#include "Ray.hpp"
#include "HitRecord.hpp"
#include "Vector3.hpp"

namespace RayTracing
{

class Material
{
public:
    virtual bool scatter(
        const Ray &ray,
        const HitRecord &hitRecord,
        Color &attenuation,
        Ray &scattered
    ) const = 0;
};

} // namespace RayTracing
