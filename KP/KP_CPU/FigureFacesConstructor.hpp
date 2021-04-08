#pragma once

#include <vector>
#include <cassert>

#include "Vector3.hpp"
#include "SquareFace.hpp"
#include "TriangleFace.hpp"

namespace RayTracing
{

enum class FigureId
{
    Cube,
    TexturedCube,
    Floor,
    FancyCube
};

class FigureFacesConstructor
{
public:
    template<FigureId figureId, typename Face>
    static void ConstructFigureFaces(
        std::vector<Face> &faces,
        std::vector<int> &facesMaterialIds,
        const Point3 &origin,
        const float radius
    )
    {
        assert(("Not implemented", false));
    }
};

} // namespace RayTracing
