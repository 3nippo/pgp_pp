#pragma once

#include "Vector3.cuh.cu"

namespace RayTracing 
{

class Ray
{
public:
    Point3 origin;
    Vector3 direction;

public:
    __host__ __device__
    Ray() {}
    __host__ __device__
    Ray(const Point3 &origin, const Vector3 &direction) : origin(origin), direction(direction) {}

    __host__ __device__
    Point3 At(const float t) const
    {
        return origin + t * direction;
    }
};

} // namespace RayTracing
