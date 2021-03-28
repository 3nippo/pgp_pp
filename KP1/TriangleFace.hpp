#pragma once

#include "Face.hpp"
#include "Vector3.hpp"
#include "Ray.hpp"


namespace RayTracing 
{

class TriangleFace : public Face
{
private:
    Point3 m_A, m_B, m_C;
    Vector3 m_normal;
    float m_D;

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
    const;

    float PlanePoint(const Ray &ray) const;
};

} // namespace RayTracing
