#pragma once

#include "Plane.cuh.cu"
#include "Vector3.cuh.cu"
#include "Ray.cuh.cu"
#include "HitRecord.cuh.cu"
#include "Texture.cuh.cu"
#include "Material.cuh.cu"
#include "aabb.cuh.cu"

namespace RayTracing 
{


class TriangleFace : public Plane
{
private:
    const Material * const * const m_material;
public:
    TriangleFace(
        const Point3 &A, 
        const Point3 &B,
        const Point3 &C,
        const Point3 &origin,
        const Material * const * const material=nullptr
    );
    
    __host__ __device__
    bool Hit(
        const Ray &ray, 
        const float tMin,
        HitRecord &hitRecord
    ) const;

    __host__ 
    aabb BoundingBox() const;
};

class MappedTriangleFace : public TriangleFace
{
private:
    TriangleMapping m_mapping;

public:
    MappedTriangleFace(
        const Point3 &A, 
        const Point3 &B,
        const Point3 &C,
        const Point3 &origin,
        const Material * const * const material,
        const TriangleMapping &mapping
    );

    MappedTriangleFace(
        const TriangleFace &triangleFace
    )
        : TriangleFace(triangleFace),
          m_mapping(
            TriangleMapping(Vector3(), Vector3(), Vector3())
          )
    {}

    __host__ __device__
    bool Hit(
        const Ray &ray,
        const float tMin,
        HitRecord &hitRecord
    ) const;
};

} // namespace RayTracing
