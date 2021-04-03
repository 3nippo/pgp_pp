#include "Vector3.hpp"

#include <cmath>

namespace RayTracing
{

Vector3 Vector3::operator-() const
{
    return Vector3{-x, -y, -z};
}

Vector3& Vector3::operator+=(const Vector3 &v)
{
    x += v.x;
    y += v.y;
    z += v.z;

    return *this;
}

Vector3 Vector3::operator+(const Vector3 &v) const
{
    return Vector3{x + v.x, y + v.y, z + v.z};
}

Vector3& Vector3::operator-=(const Vector3 &v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;

    return *this;
}

Vector3 Vector3::operator-(const Vector3 &v) const
{
    return Vector3{x - v.x, y - v.y, z - v.z};
}

/* Vector3& Vector3::operator*=(const Vector3 &v) */
/* { */
/*     x *= v.x; */
/*     y *= v.y; */
/*     z *= v.z; */

/*     return *this; */
/* } */

/* Vector3 Vector3::operator*(const Vector3 &v) const */
/* { */
/*     return Vector3{x * v.x, y * v.y, z * v.z}; */
/* } */

Vector3& Vector3::operator*=(const float t)
{
    x *= t;
    y *= t;
    z *= t;

    return *this;
}

Vector3 Vector3::operator*(const float t) const
{
    return Vector3{x * t, y * t, z * t};
}

/* Vector3& Vector3::operator/=(const Vector3 &v) */
/* { */
/*     x /= v.x; */
/*     y /= v.y; */
/*     z /= v.z; */

/*     return *this; */
/* } */

/* Vector3 Vector3::operator/(const Vector3 &v) const */
/* { */
/*     return Vector3{x / v.x, y / v.y, z / v.z}; */
/* } */

Vector3& Vector3::operator/=(const float t)
{
    x /= t;
    y /= t;
    z /= t;

    return *this;
}

Vector3 Vector3::operator/(const float t) const
{
    return Vector3{x / t, y / t, z / t};
}

float Vector3::Length() const
{
    return sqrtf(LengthSquared());   
}

float Vector3::LengthSquared() const
{
    return x*x + y*y + z*z;
}

float Vector3::Dot(const Vector3 &v) const
{
    return x * v.x + y * v.y + z * v.z;
}

Vector3 Vector3::Cross(const Vector3 &v) const
{
    return Vector3
    {
        y * v.z - z * v.y,
        z * v.x - x * v.z,
        x * v.y - y * v.x
    };
}

float Vector3::Dist(const Vector3 &v) const
{
    return (*this - v).Length();
}

Vector3 Vector3::UnitVector() const
{
    return *this / Length();
}

Vector3 operator*(float t, const Vector3 &v)
{
    return Vector3{t * v.x, t * v.y, t * v.z};
}

std::ostream& operator<<(std::ostream &stream, const Color &v)
{
    const unsigned char r = static_cast<unsigned char>(255.999f * v.x),
                        g = static_cast<unsigned char>(255.999f * v.y),
                        b = static_cast<unsigned char>(255.999f * v.z),
                        a = 255;

    stream.write(reinterpret_cast<const char*>(&r), sizeof(char));
    stream.write(reinterpret_cast<const char*>(&g), sizeof(char));
    stream.write(reinterpret_cast<const char*>(&b), sizeof(char));
    stream.write(reinterpret_cast<const char*>(&a), sizeof(char));

    return stream;
}

} // namespace RayTracing
