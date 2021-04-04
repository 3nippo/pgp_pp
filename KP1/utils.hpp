#pragma once

#include <limits>

namespace RayTracing
{

constexpr float INF = std::numeric_limits<float>::infinity();

float DegreesToRadians(const float degrees);

float Clamp(
    const float x, 
    const float xMin, 
    const float xMax
);

// uniform distribution
float GenRandom(
    const float a=0, 
    const float b=1
);

} // namespace RayTracing
