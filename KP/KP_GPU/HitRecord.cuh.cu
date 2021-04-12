#pragma once

#include "Vector3.cuh.cu"
#include "Ray.cuh.cu"

namespace RayTracing
{

class Material;

struct HitRecord
{
    float t;
    // or barycentric alpha
    float u;
    // or barycentric beta
    float v;
    Vector3 normal;
    Point3 point;
    const Material *material;
    
    __device__
    void SetNormal(const Ray &ray, const Vector3 &pointNormal)
    {
        normal = pointNormal.Dot(ray.direction) < 0 ? pointNormal : -pointNormal;
    }
};

} // namespace RayTracing
