#include "TriangleFace.cuh.cu"

#include <algorithm>


namespace RayTracing 
{
TriangleFace::TriangleFace(
    const Point3 &A, 
    const Point3 &B,
    const Point3 &C,
    const Point3 &origin,
    Material * const * const material
) : Plane(A, B, C, origin),
    m_material(material)
{}

__host__ __device__
bool TriangleFace::Hit(
    const Ray &ray, 
    const float tMin,
    HitRecord &hitRecord
) 
const
{
    float t = PlanePoint(ray);

    if (t < tMin || t > hitRecord.t)
        return false;
    
    Point3 P = ray.At(t);

    Vector3 n = (m_B - m_A).Cross(m_C - m_A);

    float alpha = n.Dot((m_C - m_B).Cross(P - m_B)) / n.LengthSquared(),
          beta = n.Dot((m_A - m_C).Cross(P - m_C)) / n.LengthSquared();

    if (alpha >= 0 && beta >= 0 && 1 - alpha - beta >= 0)
    {
        hitRecord.t = t;
        hitRecord.u = alpha;
        hitRecord.v = beta;
        hitRecord.SetNormal(ray, m_normal);
        hitRecord.point = ray.At(t);
        hitRecord.material = *m_material;

        return true;
    }

    return false;
}

__host__ 
aabb TriangleFace::BoundingBox() const
{
    return aabb{
        Point3{
            std::min({ m_A.d.x, m_B.d.x, m_C.d.x }),
            std::min({ m_A.d.y, m_B.d.y, m_C.d.y }),
            std::min({ m_A.d.z, m_B.d.z, m_C.d.z })
        },
        Point3{
            std::max({ m_A.d.x, m_B.d.x, m_C.d.x }),
            std::max({ m_A.d.y, m_B.d.y, m_C.d.y }),
            std::max({ m_A.d.z, m_B.d.z, m_C.d.z })
        }
    };
}

MappedTriangleFace::MappedTriangleFace(
    const Point3 &A, 
    const Point3 &B,
    const Point3 &C,
    const Point3 &origin,
    Material * const * const material,
    const TriangleMapping &mapping
)
    : TriangleFace(A, B, C, origin, material),
      m_mapping(mapping)
{}

__host__ __device__ 
bool MappedTriangleFace::Hit(
    const Ray &ray, 
    const float tMin,
    HitRecord &hitRecord
) const 
{
    if (TriangleFace::Hit(ray, tMin, hitRecord))
    {
        hitRecord.u = hitRecord.u * m_mapping.m_A.d.x + hitRecord.v * m_mapping.m_B.d.x + (1 - hitRecord.u - hitRecord.v) * m_mapping.m_C.d.x;
        hitRecord.v = hitRecord.u * m_mapping.m_A.d.y + hitRecord.v * m_mapping.m_B.d.y + (1 - hitRecord.u - hitRecord.v) * m_mapping.m_C.d.y;

        return true;
    }

    return false;
}

} // namespace RayTracing
