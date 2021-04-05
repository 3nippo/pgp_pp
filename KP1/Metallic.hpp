#pragma once

#include "Material.hpp"
#include "HitRecord.hpp"
#include "Ray.hpp"
#include "utils.hpp"

namespace RayTracing
{

class Metallic : public Material
{
public:
    using Material::Material;
private:
    static Vector3 RandomVectorInUnitSphere()
    {
        while (true)
        {
            const Vector3 v = Vector3::Random(-1, 1);

            if (v.LengthSquared() >= 1)
                continue;

            return v;
        }
    }

    static Vector3 MetallicRandomVector() 
    {
        return RandomVectorInUnitSphere().UnitVector();
    }

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
        attenuation = m_albedo->GetColor(hitRecord.u, hitRecord.v);

        return hitRecord.normal.Dot(scattered.direction) > 0;
    }
};

} // namespace RayTracing
