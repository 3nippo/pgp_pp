#pragma once

#include "Vector3.hpp"

namespace RayTracing
{

struct HitRecord
{
    float t;
    Vector3 normal;
};

} // namespace RayTracing
