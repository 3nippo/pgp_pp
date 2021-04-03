#pragma once

#include "Plane.hpp"
#include "Vector3.hpp"
#include "Ray.hpp"


namespace RayTracing 
{

class TriangleFace : public Plane
{
public:
    TriangleFace(
        const Point3 &planeCenter,
        const Point3 &origin
    );

    bool Hit(
        const Ray &ray, 
        const float tMin,
        const float tMax,
        float &tOutput
    ) const;
};

} // namespace RayTracing
