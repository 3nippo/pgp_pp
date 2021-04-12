#pragma once

#include "Vector3.cuh.cu"
#include "Ray.cuh.cu"

namespace RayTracing
{

class Plane
{
protected:
    Point3 m_A, m_B, m_C;
    Vector3 m_normal;
    float m_D;
    
public:
    Plane(
        const Vector3 &A,
        const Vector3 &B,
        const Vector3 &C,
        const Point3 &origin
    )
        : m_A(A + origin), m_B(B + origin), m_C(C + origin)
    {
        m_normal = (m_B - m_A).Cross((m_C - m_A)).UnitVector();
        
        m_D = m_A.Dot(m_normal);
    }
    
    __device__
    float PlanePoint(const Ray &ray) const
    {
        return (m_D - m_normal.Dot(ray.origin)) / m_normal.Dot(ray.direction);
    }

    const Vector3& GetNormal() const
    {
        return m_normal;
    }
};

} // namespace RayTracing
