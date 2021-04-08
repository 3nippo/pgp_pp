#pragma once

#include <array>

#include "Vector3.hpp"
#include "Ray.hpp"
#include "TriangleFace.hpp"
#include "HitRecord.hpp"
#include "Texture.hpp"


namespace RayTracing 
{

class SquareFace
{
protected:
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
        HitRecord &hitRecord
    ) 
    const;
};

class MappedSquareFace : public SquareFace, public SquareMapping
{
public:
    MappedSquareFace(
        const Point3 &A, 
        const Point3 &B,
        const Point3 &C,
        const Point3 &D,
        const Point3 &origin,
        const TriangleMapping &m1,
        const TriangleMapping &m2
    )
        : SquareFace(A, B, C, D, origin),
          SquareMapping(m1, m2)
    {}

    bool Hit(
        const Ray &ray, 
        const float tMin,
        const float tMax,
        HitRecord &hitRecord
    ) 
    const;
};

} // namespace RayTracing
