#pragma once

#include <vector>

#include "Figure.hpp"

namespace RayTracing
{

class CubeFacesConstructor
{
public:
    static void ConstructFaces(
        const std::vector<Face> &faces,
        const Point3 &origin,
        const float radius
    );
};

} // namespace RayTracing