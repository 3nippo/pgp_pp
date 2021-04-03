#include "FigureFacesConstructor.hpp"
#include "SquareFace.hpp"
#include <cmath>

namespace RayTracing
{

template<>
void FigureFacesConstructor::ConstructFigureFaces<>
(
    std::vector<SquareFace> &faces,
    const Point3 &origin,
    const float radius
)
{
    float halfA = radius / sqrtf(3);
    
    // front face
    faces.emplace_back(
        origin + Vector3{ 0, 0, +halfA },
        origin
    );

    // back face
    faces.emplace_back(
        origin + Vector3{ 0, 0, -halfA },
        origin
    );

    // left face
    faces.emplace_back(
        origin + Vector3{ -halfA, 0, 0 },
        origin
    );
    
    // right
    faces.emplace_back(
        origin + Vector3{ +halfA, 0, 0 },
        origin
    );
    
    // bottom face
    faces.emplace_back(
        origin + Vector3{ 0, -halfA, 0 },
        origin
    );

    // top face
    faces.emplace_back(
        origin + Vector3{ 0, +halfA, 0 },
        origin
    );
}

} // namespace RayTracing
