#pragma once

#include "Vector3.hpp"
#include "Ray.hpp"

namespace RayTracing 
{
constexpr float eps = 1e-10f;

class Face
{
public:
    virtual bool Hit(
        const Ray &ray, 
        const float tMin,
        const float tMax,
        float &tOutput
    ) 
    const = 0;
};

} // namespace RayTracing
