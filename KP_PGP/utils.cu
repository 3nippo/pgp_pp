#include "utils.cuh.cu"
#include <random>

namespace RayTracing
{

float DegreesToRadians(const float degrees)
{
    return degrees * M_PI / 180;
}

__host__ __device__
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

float GenRandom()
{
    static std::uniform_real_distribution<float> distribution(0, 1);

    static std::mt19937 generator;

    return distribution(generator);
}

float GenRandom(
    const float a,
    const float b
)
{
    return a + (b - a) * GenRandom();
}
} // namespace RayTracing
