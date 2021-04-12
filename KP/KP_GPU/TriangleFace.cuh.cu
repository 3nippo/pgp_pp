#pragma once

#include "Plane.cuh.cu"
#include "Vector3.cuh.cu"
#include "Ray.cuh.cu"
#include "HitRecord.cuh.cu"

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
    
    __device__
    bool Hit(
        const Ray &ray, 
        const float tMin,
        const float tMax,
        HitRecord &hitRecord
    ) const;
};

} // namespace RayTracing
