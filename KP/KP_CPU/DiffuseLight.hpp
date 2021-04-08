#pragma once

#include "Material.hpp"
#include "Vector3.hpp"
#include "Texture.hpp"

namespace RayTracing
{

class DiffuseLight : public Material
{
protected:
    const Texture* const m_emitterTexture;
public:
    DiffuseLight(const Texture* const emitterMaterial)
        : Material(0, 0),
          m_emitterTexture(emitterMaterial)
    {}

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
        return m_emitterTexture->GetColor(hitRecord.u, hitRecord.v);
    }
};

} // namespace RayTracing
