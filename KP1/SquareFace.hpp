#pragma once

#include "Plane.hpp"
#include "Vector3.hpp"
#include "Ray.hpp"


namespace RayTracing 
{

class SquareFace : public Plane
{
public:
    SquareFace(
        const Point3 &A, 
        const Point3 &B,
        const Point3 &C
    );

    bool Hit(
        const Ray &ray, 
        const float tMin,
        const float tMax,
        float &tOutput
    ) 
    const;
};

} // namespace RayTracing
