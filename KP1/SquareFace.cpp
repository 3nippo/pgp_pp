#include "SquareFace.hpp"
#include "utils.hpp"

namespace RayTracing 
{
SquareFace::SquareFace(
    const Point3 &A, 
    const Point3 &B,
    const Point3 &C,
    const Point3 &D,
    const Point3 &origin
)
    : m_triangleFaces({
        TriangleFace(A, B, C, origin),
        TriangleFace(A, D, C, origin)
      })
{}

bool SquareFace::Hit(
    const Ray &ray, 
    const float tMin,
    const float tMax,
    HitRecord &hitRecord
) 
const
{
    for (size_t i = 0; i < m_triangleFaces.size(); ++i)
        if (m_triangleFaces[i].Hit(ray, tMin, hitRecord.t, hitRecord))
            return true;

    return false;
}

} // namespace RayTracing
