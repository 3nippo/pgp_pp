#pragma once

#include <limits>

namespace RayTracing
{

constexpr float c_inf = std::numeric_limits<float>::infinity();

float DegreesToRadiand(const float degrees);

} // namespace RayTracing
