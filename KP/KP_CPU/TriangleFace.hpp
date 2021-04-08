#pragma once

#include "Plane.hpp"
#include "Vector3.hpp"
#include "Ray.hpp"
#include "HitRecord.hpp"

namespace RayTracing 
{


class TriangleFace : public Plane
{
public:
    TriangleFace(
        const Point3 &A, 
        const Point3 &B,
        const Point3 &C,
        const Point3 &origin
    );

    bool Hit(
        const Ray &ray, 
        const float tMin,
        const float tMax,
        HitRecord &hitRecord
    ) const;
};

} // namespace RayTracing
