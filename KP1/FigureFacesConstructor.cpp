#include "FigureFacesConstructor.hpp"
#include "SquareFace.hpp"
#include <cmath>

namespace RayTracing
{

template<>
void FigureFacesConstructor::ConstructFigureFaces
(
    std::vector<SquareFace> &faces,
    const Point3 &origin,
    const float radius
)
{
    float halfA = radius / sqrtf(3);
    
    // front face
    faces.emplace_back(
        Point3{ -halfA, -halfA, +halfA },
        Point3{ +halfA, -halfA, +halfA },
        Point3{ +halfA, +halfA, +halfA },
        Point3{ -halfA, +halfA, +halfA },
        origin
    );

    // back face
    /* faces.emplace_back( */
    /*     Point3{ -halfA, -halfA, -halfA }, */
    /*     Point3{ +halfA, -halfA, -halfA }, */
    /*     Point3{ +halfA, +halfA, -halfA }, */
    /*     origin */
    /* ); */

    /* // left face */
    /* faces.emplace_back( */
    /*     Point3{ -halfA, -halfA, +halfA }, */
    /*     Point3{ -halfA, -halfA, -halfA }, */
    /*     Point3{ -halfA, +halfA, -halfA }, */
    /*     origin */
    /* ); */
    
    /* // right */
    /* faces.emplace_back( */
    /*     Point3{ +halfA, -halfA, +halfA }, */
    /*     Point3{ +halfA, -halfA, -halfA }, */
    /*     Point3{ +halfA, +halfA, -halfA }, */
    /*     origin */
    /* ); */
    
    /* // bottom face */
    /* faces.emplace_back( */
    /*     Point3{ +halfA, -halfA, +halfA }, */
    /*     Point3{ +halfA, -halfA, -halfA }, */
    /*     Point3{ -halfA, -halfA, -halfA }, */
    /*     origin */
    /* ); */
}

} // namespace RayTracing
