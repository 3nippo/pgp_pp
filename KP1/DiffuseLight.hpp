#pragma once

#include "Material.hpp"
#include "Vector3.hpp"

namespace RayTracing
{

class DiffuseLight : public Material
{
public:
    using Material::Material;
private:
    virtual bool Scatter(
        const Ray &ray,
        const HitRecord &hitRecord,
        Color &attenuation,
        Ray &scattered
    ) const override
    {
        return false;
    }


    virtual Color Emitted(
        const HitRecord &hitRecord
    ) const override
    {
        return m_albedo->GetColor(hitRecord.u, hitRecord.v);
    }
};

} // namespace RayTracing
