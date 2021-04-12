#pragma once

#include "Material.cuh.cu"
#include "Vector3.cuh.cu"
#include "Texture.cuh.cu"

namespace RayTracing
{

class DiffuseLight : public Material
{
protected:
    const Texture* const* const m_emitterTexture;
public:
    __host__ __device__
    DiffuseLight(const Texture* const* const emitterMaterial)
        : Material(0, 0),
          m_emitterTexture(emitterMaterial)
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
        return false;
    }

    __device__
    virtual Color Emitted(
        const HitRecord &hitRecord
    ) const override
    {
        return (*m_emitterTexture)->GetColor(hitRecord.u, hitRecord.v);
    }

    __device__
    virtual bool Emits() const override
    {
        return true;
    }
};

} // namespace RayTracing
