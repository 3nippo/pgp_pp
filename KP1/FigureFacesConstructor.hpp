#pragma once

#include <vector>

#include "Vector3.hpp"
#include "SquareFace.hpp"
#include "TriangleFace.hpp"

namespace RayTracing
{

class FigureFacesConstructor
{
public:
    template<typename Face>
    static void ConstructFigureFaces(
        std::vector<Face> &faces,
        const Point3 &origin,
        const float radius
    )
    {
        static_assert(false, "Not implemented");
    }

    template<>
    void ConstructFigureFaces<>
    (
        std::vector<SquareFace> &faces,
        const Point3 &origin,
        const float radius
    );
};

} // namespace RayTracing
