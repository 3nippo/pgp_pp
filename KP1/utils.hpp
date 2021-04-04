#pragma once

#include <limits>

namespace RayTracing
{

constexpr float INF = std::numeric_limits<float>::infinity();
constexpr float EPS = 1e-8;

float DegreesToRadians(const float degrees);

float Clamp(
    const float x, 
    const float xMin, 
    const float xMax
);

// uniform distribution
float GenRandom();

// uniform distribution
float GenRandom(
    const float a, 
    const float b
);

} // namespace RayTracing
