#include "FigureFacesConstructor.hpp"
#include "SquareFace.hpp"
#include <cmath>
#include "Texture.hpp"

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
    faces.emplace_back(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ -halfA, +halfA, -halfA },
        origin
    );

    // left face
    faces.emplace_back(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ -halfA, +halfA, -halfA },
        Point3{ -halfA, +halfA, +halfA },
        Point3{ -halfA, -halfA, +halfA },
        origin
    );
    
    // right face
    faces.emplace_back(
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, +halfA },
        Point3{ +halfA, -halfA, +halfA },
        origin
    );
    
    // bottom face
    faces.emplace_back(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, +halfA },
        Point3{ -halfA, -halfA, +halfA },
        origin
    );

    // top face
    faces.emplace_back(
        Point3{ -halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, +halfA },
        Point3{ -halfA, +halfA, +halfA },
        origin
    );
}

template<>
void FigureFacesConstructor::ConstructFigureFaces
(
    std::vector<MappedSquareFace> &faces,
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
        origin,
        TriangleMapping{
            Point3{ 0, 0, 0 },
            Point3{ 0, 0, 0 },
            Point3{ 0, 0, 0 }
        },
        TriangleMapping{
            Point3{ 0, 1, 0 },
            Point3{ 0, 0, 0 },
            Point3{ 1, 0, 0 }
        }
    );

    // back face
    faces.emplace_back(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ -halfA, +halfA, -halfA },
        origin,
        TriangleMapping{
            Point3{ 1, 1, 0 },
            Point3{ 0, 1, 0 },
            Point3{ 0, 0, 0 }
        },
        TriangleMapping{
            Point3{ 1, 1, 0 },
            Point3{ 1, 0, 0 },
            Point3{ 0, 0, 0 }
        }
    );

    // left face
    faces.emplace_back(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ -halfA, +halfA, -halfA },
        Point3{ -halfA, +halfA, +halfA },
        Point3{ -halfA, -halfA, +halfA },
        origin,
        TriangleMapping{
            Point3{ 0, 1, 0 },
            Point3{ 0, 0, 0 },
            Point3{ 1, 0, 0 }
        },
        TriangleMapping{
            Point3{ 0, 1, 0 },
            Point3{ 1, 1, 0 },
            Point3{ 1, 0, 0 }
        }
    );
    
    // right face
    faces.emplace_back(
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, +halfA },
        Point3{ +halfA, -halfA, +halfA },
        origin,
        TriangleMapping{
            Point3{ 1, 1, 0 },
            Point3{ 1, 0, 0 },
            Point3{ 0, 0, 0 }
        },
        TriangleMapping{
            Point3{ 1, 1, 0 },
            Point3{ 0, 1, 0 },
            Point3{ 0, 0, 0 }
        }
    );
    
    // bottom face
    faces.emplace_back(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, +halfA },
        Point3{ -halfA, -halfA, +halfA },
        origin,
        TriangleMapping{
            Point3{ 0, 1, 0 },
            Point3{ 1, 1, 0 },
            Point3{ 1, 0, 0 }
        },
        TriangleMapping{
            Point3{ 0, 1, 0 },
            Point3{ 0, 0, 0 },
            Point3{ 1, 0, 0 }
        }
    );

    // top face
    faces.emplace_back(
        Point3{ -halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, +halfA },
        Point3{ -halfA, +halfA, +halfA },
        origin,
        TriangleMapping{
            Point3{ 0, 1, 0 },
            Point3{ 1, 1, 0 },
            Point3{ 1, 0, 0 }
        },
        TriangleMapping{
            Point3{ 0, 1, 0 },
            Point3{ 0, 0, 0 },
            Point3{ 1, 0, 0 }
        }
    );
}
} // namespace RayTracing
