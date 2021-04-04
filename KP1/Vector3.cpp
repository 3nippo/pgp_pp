#include "Vector3.hpp"

#include <cmath>

#include "utils.hpp"

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

void ColorToRGBA(
    const Color &color,
    unsigned char &r,
    unsigned char &g,
    unsigned char &b,
    unsigned char &a
)
{
    r = static_cast<unsigned char>(255.999f * color.x);
    g = static_cast<unsigned char>(255.999f * color.y);
    b = static_cast<unsigned char>(255.999f * color.z);
    a = 255;
}

Vector3 Vector3::Random(const float a, const float b)
{
    return Vector3{
        GenRandom(a, b),
        GenRandom(a, b),
        GenRandom(a, b)
    };   
}

std::ostream& operator<<(std::ostream &stream, const Color &v)
{
    unsigned char r, g, b, a;

    ColorToRGBA(v, r, g, b, a);

    stream.write(reinterpret_cast<const char*>(&r), sizeof(char));
    stream.write(reinterpret_cast<const char*>(&g), sizeof(char));
    stream.write(reinterpret_cast<const char*>(&b), sizeof(char));
    stream.write(reinterpret_cast<const char*>(&a), sizeof(char));

    return stream;
}

void Vector3::Clamp(const float tMin, const float tMax)
{
    x = RayTracing::Clamp(x, tMin, tMax);
    y = RayTracing::Clamp(y, tMin, tMax);
    z = RayTracing::Clamp(z, tMin, tMax);
}

} // namespace RayTracing
