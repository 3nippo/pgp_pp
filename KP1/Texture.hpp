#pragma once

#include "Vector3.hpp"
#include <array>

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

class TriangleMapping
{
public:
    Vector3 m_A;
    Vector3 m_B;
    Vector3 m_C;
};

class SquareMapping
{
protected:
    std::array<TriangleMapping, 2> m_triangleMappings;
public:
    SquareMapping(const TriangleMapping &m1, const TriangleMapping &m2)
        : m_triangleMappings({ m1, m2 })
    {}
};
}; // namespace RayTracing
