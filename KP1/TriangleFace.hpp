#pragma once

#include "Face.hpp"
#include "Plane.hpp"
#include "Vector3.hpp"
#include "Ray.hpp"


namespace RayTracing 
{

class TriangleFace : public Face, Plane
{
public:
    TriangleFace(
        const Point3 &A, 
        const Point3 &B,
        const Point3 &C
    );

private:
    virtual bool Hit(
        const Ray &ray, 
        const float tMin,
        const float tMax,
        float &tOutput
    ) 
    const override;
};

} // namespace RayTracing
