#pragma once

#include "Material.cuh.cu"
#include "HitRecord.cuh.cu"
#include "Ray.cuh.cu"
#include "utils.cuh.cu"

namespace RayTracing
{

class Metallic : public Material
{
protected:
    const Texture* const * const m_albedo;
public:
    __host__ __device__
    Metallic(
        const float transparency,
        const float reflectance,
        const Texture* const * const albedo
    )
        : Material(transparency, reflectance),
          m_albedo(albedo)
    {}
private:
    __device__
    virtual bool Scatter(
        const Ray &ray,
        const HitRecord &hitRecord,
        Color &attenuation,
        Ray &scattered
    ) const override
    {
        const Vector3 reflected = Vector3::Reflect(
            ray.direction.UnitVector(), 
            hitRecord.normal
        );

        scattered = Ray(hitRecord.point, reflected);
        attenuation = (*m_albedo)->GetColor(hitRecord.u, hitRecord.v);

        return hitRecord.normal.Dot(scattered.direction) > 0;
    }
};

} // namespace RayTracing
