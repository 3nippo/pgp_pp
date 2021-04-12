#include "TriangleFace.cuh.cu"


namespace RayTracing 
{
TriangleFace::TriangleFace(
    const Point3 &A, 
    const Point3 &B,
    const Point3 &C,
    const Point3 &origin
) : Plane(A, B, C, origin)
{}

__device__
bool TriangleFace::Hit(
    const Ray &ray, 
    const float tMin,
    const float tMax,
    HitRecord &hitRecord
) 
const
{
    float t = PlanePoint(ray);

    if (t < tMin || t > tMax)
        return false;
    
    Point3 P = ray.At(t);

    Vector3 n = (m_B - m_A).Cross(m_C - m_A);

    float alpha = n.Dot((m_C - m_B).Cross(P - m_B)) / n.LengthSquared(),
          beta = n.Dot((m_A - m_C).Cross(P - m_C)) / n.LengthSquared();

    float gamma = 1 - alpha - beta;

    if (alpha >= 0 && beta >= 0 && gamma >= 0)
    {
        hitRecord.t = t;
        hitRecord.u = alpha;
        hitRecord.v = beta;
        hitRecord.SetNormal(ray, m_normal);
        hitRecord.point = ray.At(t);

        return true;
    }

    return false;
}

} // namespace RayTracing