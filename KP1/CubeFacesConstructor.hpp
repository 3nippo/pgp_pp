#pragma once

#include <vector>

#include "Vector3.hpp"
#include "SquareFace.hpp"
#include "TriangleFace.hpp"

namespace RayTracing
{

enum class FigureId
{

Cube 

}; // enum class Figure

class FigureFacesConstructor
{
public:
    template<FigureId figureId, typename Face>
    static void ConstructFigureFaces(
        std::vector<Face> &faces,
        const Point3 &origin,
        const float radius
    )
    {
        static_assert(false, "Not implemented");
    }

    template<>
    void ConstructFigureFaces<FigureId::Cube>
    (
        std::vector<SquareFace> &faces,
        const Point3 &origin,
        const float radius
    );
};

} // namespace RayTracing
