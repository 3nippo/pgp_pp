#pragma once

#include "Ray.hpp"
#include "HitRecord.hpp"
#include "Vector3.hpp"

namespace RayTracing
{

class Material
{
protected:
    Color m_albedo;
public:
    Material(const Color &albedo)
        : m_albedo(albedo)
    {}

    virtual bool scatter(
        const Ray &ray,
        const HitRecord &hitRecord,
        Color &attenuation,
        Ray &scattered
    ) const = 0;
};

} // namespace RayTracing
