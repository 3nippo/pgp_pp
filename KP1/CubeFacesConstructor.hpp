#pragma once

#include <vector>

#include "Figure.hpp"

namespace RayTracing
{

class CubeFacesConstructor
{
public:
    template<typename Face>
    static void ConstructFaces(
        std::vector<Face> &faces,
        const Point3 &origin,
        const float radius
    );
};

} // namespace RayTracing
