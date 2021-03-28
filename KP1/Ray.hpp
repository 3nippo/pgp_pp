#pragma once

#include "Vector3.hpp"

namespace RayTracing 
{

class Ray
{
public:
    Point3 origin;
    Vector3 direction;

public:
    Ray(const Point3 &origin, const Vector3 &direction) : origin(origin), direction(direction) {}

    Point3 At(const float t) const
    {
        return origin + t * direction;
    }
};

} // namespace RayTracing
