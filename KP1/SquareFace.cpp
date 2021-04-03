#include "SquareFace.hpp"


namespace RayTracing 
{
SquareFace::SquareFace(
    const Point3 &planeCenter,
    const Point3 &origin
) : Plane(planeCenter, origin)
{}

bool SquareFace::Hit(
    const Ray &ray, 
    const float tMin,
    const float tMax,
    float &tOutput
) 
const
{
    float t = PlanePoint(ray);

    if (t < tMin || t > tMax)
        return false;
    
    Point3 P = ray.At(t);

    float area = (m_B - m_A).Cross((m_C - m_A)).Length() / 2;

    float alpha = (m_B - P).Cross((m_C - P)).Length() / 2 / area;

    float beta = (m_C - P).Cross((m_A - P)).Length() / 2 / area;

    if (alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1)
    {
        tOutput = t;

        return true;
    }

    return false;
}

} // namespace RayTracing
