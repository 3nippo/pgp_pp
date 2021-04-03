#pragma once

#include <vector>
#include <cassert>

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
        assert(("Not implemented", false));
    }
};

} // namespace RayTracing
