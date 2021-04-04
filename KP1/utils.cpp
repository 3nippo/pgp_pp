#include <cmath>
#include <random>

#include "utils.hpp"


namespace RayTracing
{

float DegreesToRadians(const float degrees)
{
    return degrees * M_PI / 180;
}

float Clamp(
    const float x, 
    const float xMin, 
    const float xMax
)
{
    if (x < xMin)
        return xMin;

    if (x > xMax)
        return xMax;

    return x;
}

float GenRandom(
    const float a,
    const float b
)
{
    static std::uniform_real_distribution<float> distribution(a, b);

    static std::mt19937 generator;

    return distribution(generator);
}

} // namespace RayTracing
