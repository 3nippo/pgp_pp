#pragma once

#include "Vector3.cuh.cu"
#include <array>
#include "dummy_helper.cuh.cu"

namespace RayTracing
{

class Texture
{
public:
    __device__
    virtual Color GetColor(const float u, const float v) const = 0;

    __device__
    virtual ~Texture() {}
};

class SolidTexture : public Texture
{
private:
    Color m_color;
public:
    __device__
    SolidTexture(const Color &color)
        : m_color(color)
    {}

    __device__
    virtual ~SolidTexture() {}

private:
    __device__
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
    TriangleMapping m_triangleMapping1;
    TriangleMapping m_triangleMapping2;
public:
    SquareMapping(const TriangleMapping &m1, const TriangleMapping &m2)
        : m_triangleMapping1(m1),
          m_triangleMapping2(m2)
    {}
};
}; // namespace RayTracing
