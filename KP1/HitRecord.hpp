#pragma once

#include "Vector3.hpp"

namespace RayTracing
{

class Material;

struct HitRecord
{
    float t;
    Vector3 normal;
    Material *material;
};

} // namespace RayTracing
