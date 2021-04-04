#pragma once

#include "Material.hpp"
#include "HitRecord.hpp"
#include "Ray.hpp"
#include "utils.hpp"

namespace RayTracing
{

class Lambertian : public Material
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

    static Vector3 LambertianRandomVector() 
    {
        return RandomVectorInUnitSphere().UnitVector();
    }

    virtual bool scatter(
        const Ray &ray,
        const HitRecord &hitRecord,
        Color &attenuation,
        Ray &scattered
    ) const override
    {
        Vector3 scatterDir = hitRecord.normal + LambertianRandomVector();

        if (scatterDir.NearZero())
            scatterDir = hitRecord.normal;

        scattered = Ray(
            hitRecord.point,
            scatterDir
        );

        attenuation = m_albedo->GetColor(hitRecord.u, hitRecord.v);

        return true;
    }
};

} // namespace RayTracing
