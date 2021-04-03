#pragma once

#include "Vector3.hpp"
#include "Ray.hpp"

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
        const Vector3 &C
    )
        : m_A(A), m_B(B), m_C(C)
    {
        m_normal = (B - A).Cross((C - A)).UnitVector();

        m_D = A.Dot(m_normal);
    }

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
