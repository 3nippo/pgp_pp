#pragma once

#include "Vector3.cuh.cu"
#include <array>
#include "dummy_helper.cuh.cu"

namespace RayTracing
{

class Texture
{
public:
    __host__ __device__
    virtual Color GetColor(const float u, const float v) const
    {
        return Color();
    }

    __host__ __device__
    virtual ~Texture() {}
};

class SolidTexture : public Texture
{
private:
    Color m_color;
public:
   __host__  __device__
    SolidTexture(const Color &color)
        : m_color(color)
    {}

   __host__  __device__
    virtual ~SolidTexture() {}

private:
   __host__ __device__
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

public:
    TriangleMapping(const Vector3& A, const Vector3 &B, const Vector3 &C)
        : m_A(A), m_B(B), m_C(C)
    {}
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
