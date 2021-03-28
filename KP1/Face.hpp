#pragma once

#include "Vector3.hpp"
#include "Ray.hpp"

namespace RayTracing 
{

struct HitRecord
{
   Point3 point;
   Vector3 normal;
   float t;
};

class Face
{
public:
    virtual bool hit(
        const Ray &ray, 
        const float tMin,
        const float tMax,
        HitRecord &hitRecord
    ) 
    const = 0;
};

} // namespace RayTracing
