#pragma once

#include <iostream>

namespace RayTracing
{

class Vector3
{
public:
    float4 d;

public:
    __host__ __device__
    Vector3() : d({ 0, 0, 0, 0}) {}

    __host__ __device__
    Vector3(float x, float y, float z, float w=0) : d({ x, y, z, w }) {}

    __host__ __device__
    Vector3(const float4 &v) : d(v) {}

    __host__ __device__
    Vector3 operator-() const;

    __host__ __device__
    Vector3& operator+=(const Vector3 &v);
    __host__ __host__ __device__
    Vector3 operator+(const Vector3 &v) const;

    __host__ __device__
    Vector3& operator-=(const Vector3 &v);
    __host__ __device__
    Vector3 operator-(const Vector3 &v) const;

    __host__ __device__
    Vector3& operator*=(const Vector3 &v);
    __host__ __device__
    Vector3 operator*(const Vector3 &v) const;
    /* __host__ __device__ */
    /* Vector3 operator*(const float4 &v) const; */
    __host__ __device__
    Vector3& operator*=(const float t);
    __host__ __device__
    Vector3 operator*(const float t) const;

    __host__ __device__
    Vector3& operator/=(const Vector3 &v) = delete;
    __host__ __device__
    Vector3 operator/(const Vector3 &v) const = delete;
    __host__ __device__
    Vector3& operator/=(const float t);
    __host__ __device__
    Vector3 operator/(const float t) const;

    __host__ __device__
    float Length() const;

    __host__ __device__
    float LengthSquared() const;

    __host__ __device__
    float Dot(const Vector3 &v) const;
    __host__ __device__
    Vector3 Cross(const Vector3 &v) const;
    __host__ __device__
    float Dist(const Vector3 &v) const;
    __host__ __device__
    Vector3 UnitVector() const;
    __host__ __device__
    bool NearZero() const;

    __device__
    void AtomicExch(const Vector3& v);

    __host__ __device__
    Vector3& Clamp(const float tMin, const float tMax);
    
    __host__ __device__
    static Vector3 Reflect(const Vector3 &v, const Vector3 &normal);
};

__host__ __device__
Vector3 operator*(float t, const Vector3 &v);

using Point3 = Vector3;

// RGB color
using Color = Vector3; 

__host__ __device__
void ColorToRGBA(
    const Color &color,
    unsigned char &r,
    unsigned char &g,
    unsigned char &b,
    unsigned char &a
);

std::ostream& operator<<(std::ostream &stream, const Color &v);
__host__ 
std::istream& operator>>(std::istream &istream, Vector3 &v);

} // namespace RayTracing
