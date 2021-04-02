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
        origin + Point3{ -halfA, -halfA, +halfA },
        origin + Point3{ +halfA, -halfA, +halfA },
        origin + Point3{ +halfA, +halfA, +halfA }
    );

    // back face
    faces.emplace_back(
        origin + Point3{ -halfA, -halfA, -halfA },
        origin + Point3{ +halfA, -halfA, -halfA },
        origin + Point3{ +halfA, +halfA, -halfA }
    );

    // left face
    faces.emplace_back(
        origin + Point3{ -halfA, -halfA, +halfA },
        origin + Point3{ -halfA, -halfA, -halfA },
        origin + Point3{ -halfA, +halfA, -halfA }
    );
    
    // right
    faces.emplace_back(
        origin + Point3{ +halfA, -halfA, +halfA },
        origin + Point3{ +halfA, -halfA, -halfA },
        origin + Point3{ +halfA, +halfA, -halfA }
    );
    
    // bottom face
    faces.emplace_back(
        origin + Point3{ +halfA, -halfA, +halfA },
        origin + Point3{ +halfA, -halfA, -halfA },
        origin + Point3{ -halfA, -halfA, -halfA }
    );
}

} // namespace RayTracing
