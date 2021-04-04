#pragma once

#include "Vector3.hpp"

namespace RayTracing
{
class Texture
{
public:
    virtual Color GetColor(const float u, const float v) const = 0;
};

class SolidTexture : Texture
{
private:
    Color m_color;
public:
    SolidTexture(const Color &color)
        : m_color(color)
    {}

private:
    virtual Color GetColor(const float u, const float v) const override
    {
        return m_color;
    }
};
}; // namespace RayTracing
