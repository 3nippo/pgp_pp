#pragma once

#include <array>

#include "Vector3.cuh.cu"
#include "Ray.cuh.cu"
#include "TriangleFace.cuh.cu"
#include "HitRecord.cuh.cu"
#include "Texture.cuh.cu"


namespace RayTracing 
{

class SquareFace
{
protected:
    TriangleFace m_triangleFace1;
    TriangleFace m_triangleFace2;
public:
    SquareFace(
        const Point3 &A, 
        const Point3 &B,
        const Point3 &C,
        const Point3 &D,
        const Point3 &origin
    );
    
    __device__
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
    
    __device__
    bool Hit(
        const Ray &ray, 
        const float tMin,
        const float tMax,
        HitRecord &hitRecord
    ) 
    const;

private:
    __device__
    void PlaceHitResult(HitRecord &hitRecord, const TriangleMapping &mapping) const;
};

} // namespace RayTracing
