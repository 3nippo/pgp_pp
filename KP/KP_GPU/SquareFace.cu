#include "SquareFace.cuh.cu"
#include "utils.cuh.cu"

namespace RayTracing 
{
SquareFace::SquareFace(
    const Point3 &A, 
    const Point3 &B,
    const Point3 &C,
    const Point3 &D,
    const Point3 &origin
)
    : m_triangleFace1({ A, B, C, origin }),
      m_triangleFace2({ A, D, C, origin })
{}

__device__
bool SquareFace::Hit(
    const Ray &ray, 
    const float tMin,
    const float tMax,
    HitRecord &hitRecord
) 
const
{
    if (m_triangleFace1.Hit(ray, tMin, hitRecord.t, hitRecord))
        return true;

    if (m_triangleFace2.Hit(ray, tMin, hitRecord.t, hitRecord))
        return true;

    return false;
}

__device__
bool MappedSquareFace::Hit(
    const Ray &ray, 
    const float tMin,
    const float tMax,
    HitRecord &hitRecord
) 
const
{
    if (m_triangleFace1.Hit(ray, tMin, hitRecord.t, hitRecord))
    {
        PlaceHitResult(hitRecord, m_triangleMapping1);
        return true;
    }

    if (m_triangleFace2.Hit(ray, tMin, hitRecord.t, hitRecord))
    {
        PlaceHitResult(hitRecord, m_triangleMapping2);
        return true;
    }

    return false;
}

__device__
void MappedSquareFace::PlaceHitResult(HitRecord &hitRecord, const TriangleMapping &mapping) const
{
    Vector3 textureCoords = \
        hitRecord.u * mapping.m_A \
        + hitRecord.v * mapping.m_B \
        + (1 - hitRecord.u - hitRecord.v) * mapping.m_C;

    hitRecord.u = textureCoords.d.x;
    hitRecord.v = textureCoords.d.y;
}


} // namespace RayTracing
