#pragma once

#include "Vector3.hpp"

namespace RayTracing
{

class Material;

struct HitRecord
{
    float t;
    Vector3 normal;
    const Material *material;
};

} // namespace RayTracing
