#include "Vector3.cuh.cu"

#include <cmath>

#include "utils.cuh.cu"

namespace RayTracing
{

__host__ __device__
Vector3 Vector3::operator-() const
{
    return Vector3{-d.x, -d.y, -d.z};
}

__host__ __device__
Vector3& Vector3::operator+=(const Vector3 &v)
{
    d.x += v.d.x;
    d.y += v.d.y;
    d.z += v.d.z;

    return *this;
}

__host__ __device__
Vector3 Vector3::operator+(const Vector3 &v) const
{
    return Vector3{d.x + v.d.x, d.y + v.d.y, d.z + v.d.z};
}

__host__ __device__
Vector3& Vector3::operator-=(const Vector3 &v)
{
    d.x -= v.d.x;
    d.y -= v.d.y;
    d.z -= v.d.z;

    return *this;
}

__host__ __device__
Vector3 Vector3::operator-(const Vector3 &v) const
{
    return Vector3{d.x - v.d.x, d.y - v.d.y, d.z - v.d.z};
}

__host__ __device__
Vector3& Vector3::operator*=(const Vector3 &v)
{
    d.x *= v.d.x;
    d.y *= v.d.y;
    d.z *= v.d.z;

    return *this;
}

__host__ __device__
Vector3 Vector3::operator*(const Vector3 &v) const
{
    return Vector3{d.x * v.d.x, d.y * v.d.y, d.z * v.d.z};
}

__host__ __device__
Vector3& Vector3::operator*=(const float t)
{
    d.x *= t;
    d.y *= t;
    d.z *= t;

    return *this;
}

__host__ __device__
Vector3 Vector3::operator*(const float t) const
{
    return Vector3{d.x * t, d.y * t, d.z * t};
}

/* __host__ __device__ */
/* Vector3 operator*(const float4 &v) const */
/* { */
/*     return Vector3{d.x * x, } */
/* } */

/* Vector3& Vector3::operator/=(const Vector3 &v) */
/* { */
/*     d.x /= v.d.x; */
/*     d.y /= v.d.y; */
/*     d.z /= v.d.z; */

/*     return *this; */
/* } */

/* Vector3 Vector3::operator/(const Vector3 &v) const */
/* { */
/*     return Vector3{d.x / v.d.x, d.y / v.d.y, d.z / v.d.z}; */
/* } */

__host__ __device__
Vector3& Vector3::operator/=(const float t)
{
    d.x /= t;
    d.y /= t;
    d.z /= t;

    return *this;
}

__host__ __device__
Vector3 Vector3::operator/(const float t) const
{
    return Vector3{d.x / t, d.y / t, d.z / t};
}

__host__ __device__
const float& Vector3::operator[](const int i) const
{
    return *(reinterpret_cast<const float*>(&d) + i);
}

__host__ __device__
float Vector3::Length() const
{
    return sqrtf(LengthSquared());   
}

__host__ __device__
float Vector3::LengthSquared() const
{
    return d.x*d.x + d.y*d.y + d.z*d.z;
}

__host__ __device__
float Vector3::Dot(const Vector3 &v) const
{
    return d.x * v.d.x + d.y * v.d.y + d.z * v.d.z;
}

__host__ __device__
Vector3 Vector3::Cross(const Vector3 &v) const
{
    return Vector3
    {
        d.y * v.d.z - d.z * v.d.y,
        d.z * v.d.x - d.x * v.d.z,
        d.x * v.d.y - d.y * v.d.x
    };
}

__host__ __device__
float Vector3::Dist(const Vector3 &v) const
{
    return (*this - v).Length();
}

__host__ __device__
Vector3 Vector3::UnitVector() const
{
    return *this / Length();
}

__host__ __device__
bool Vector3::NearZero() const
{
    return fabs(d.x) < 0.001f && fabs(d.y) < 0.001f && fabs(d.z) < 0.001f; 
    /* return fabs(d.x) == 0 && fabs(d.y) == 0 && fabs(d.z) == 0; */ 
}

__device__
void Vector3::AtomicExch(const Vector3& v)
{
    atomicExch(&d.x, v.d.x);
    atomicExch(&d.y, v.d.y);
    atomicExch(&d.z, v.d.z);
}

__host__ __device__
Vector3 operator*(float t, const Vector3 &v)
{
    return Vector3{t * v.d.x, t * v.d.y, t * v.d.z};
}

__host__ __device__
void ColorToRGBA(
    const Color &color,
    unsigned char &r,
    unsigned char &g,
    unsigned char &b,
    unsigned char &a
)
{
    r = static_cast<unsigned char>(255.999f * color.d.x);
    g = static_cast<unsigned char>(255.999f * color.d.y);
    b = static_cast<unsigned char>(255.999f * color.d.z);
    a = 255;
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

__host__ __device__
Vector3& Vector3::Clamp(const float tMin, const float tMax)
{
    d.x = RayTracing::Clamp(d.x, tMin, tMax);
    d.y = RayTracing::Clamp(d.y, tMin, tMax);
    d.z = RayTracing::Clamp(d.z, tMin, tMax);

    return *this;
}

__host__ __device__
Vector3 Vector3::Reflect(const Vector3 &v, const Vector3 &normal)
{
    return v - 2 * v.Dot(normal) * normal;
}

__host__ 
std::istream& operator>>(std::istream &istream, Vector3 &v)
{
    istream >> v.d.x >> v.d.y >> v.d.z;

    return istream;
}

} // namespace RayTracing
