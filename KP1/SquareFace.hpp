#pragma once

#include <array>

#include "Plane.hpp"
#include "Vector3.hpp"
#include "Ray.hpp"
#include "TriangleFace.hpp"


namespace RayTracing 
{

class SquareFace
{
private:
    const std::array<TriangleFace, 2> m_triangleFaces;
public:
    SquareFace(
        const Point3 &A, 
        const Point3 &B,
        const Point3 &C,
        const Point3 &D,
        const Point3 &origin
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
