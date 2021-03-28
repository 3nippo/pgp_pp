#pragma once

#include <iostream>

namespace RayTracing
{

class Vector3
{
public:
    float x, y, z;

public:
    Vector3() : x(0), y(0), z(0) {}

    Vector3(float x, float y, float z) : x(x), y(y), z(z) {}

    Vector3 operator-() const;

    Vector3& operator+=(const Vector3 &v);
    Vector3 operator+(const Vector3 &v) const;

    Vector3& operator-=(const Vector3 &v);
    Vector3 operator-(const Vector3 &v) const;

    Vector3& operator*=(const Vector3 &v) = delete;
    Vector3 operator*(const Vector3 &v) const = delete;
    Vector3& operator*=(const float t);
    Vector3 operator*(const float t) const;

    Vector3& operator/=(const Vector3 &v) = delete;
    Vector3 operator/(const Vector3 &v) const = delete;
    Vector3& operator/=(const float t);
    Vector3 operator/(const float t) const;

    float Length() const;

    float Length_squared() const;

    float Dot(const Vector3 &v) const;
    Vector3 Cross(const Vector3 &v) const;
    Vector3 UnitVector() const;
};

Vector3 operator*(float t, const Vector3 &v);

using Point3 = Vector3;

// RGB color
using Color = Vector3; 

std::ostream& operator<<(std::ostream &stream, const Color &v);

} // namespace RayTracing
