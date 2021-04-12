#pragma once

#include <limits>
#include <curand.h>
#include <curand_kernel.h>

#define INF FLT_MAX
#define EPS 1e-8
#define INT_INF INT_MAX

namespace RayTracing
{

float DegreesToRadians(const float degrees);

__device__
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
