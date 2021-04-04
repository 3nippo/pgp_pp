#pragma once

#include "Vector3.hpp"

namespace RayTracing
{

class Material;

struct HitRecord
{
    float t;
    Vector3 normal;
    Point3 point;
    const Material *material;
};

} // namespace RayTracing
