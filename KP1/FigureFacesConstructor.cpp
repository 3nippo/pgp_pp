#include "FigureFacesConstructor.hpp"
#include "SquareFace.hpp"
#include <cmath>
#include "Texture.hpp"

namespace RayTracing
{

// Given materials: Floor
template<>
void FigureFacesConstructor::ConstructFigureFaces<FigureId::Floor>
(
    std::vector<MappedSquareFace> &faces,
    std::vector<int> &facesMaterialIds,
    const Point3 &origin,
    const float radius
)
{
    float halfA = radius / sqrtf(3);

    faces.emplace_back(
        Point3{ -halfA, 0, -halfA },
        Point3{ +halfA, 0, -halfA },
        Point3{ +halfA, 0, +halfA },
        Point3{ -halfA, 0, +halfA },
        origin,
        TriangleMapping{
            Point3{ 0, 0, 0},
            Point3{ 1, 0, 0},
            Point3{ 1, 1, 0}
        },
        TriangleMapping{
            Point3{ 0, 0, 0},
            Point3{ 0, 1, 0},
            Point3{ 1, 1, 0}
        }
    );

    facesMaterialIds.push_back(0);
}

// Given materials: Face
template<>
void FigureFacesConstructor::ConstructFigureFaces<FigureId::Cube>
(
    std::vector<SquareFace> &faces,
    std::vector<int> &facesMaterialIds,
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
    facesMaterialIds.push_back(0);

    // back face
    faces.emplace_back(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ -halfA, +halfA, -halfA },
        origin
    );
    facesMaterialIds.push_back(0);

    // left face
    faces.emplace_back(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ -halfA, +halfA, -halfA },
        Point3{ -halfA, +halfA, +halfA },
        Point3{ -halfA, -halfA, +halfA },
        origin
    );
    facesMaterialIds.push_back(0);
    
    // right face
    faces.emplace_back(
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, +halfA },
        Point3{ +halfA, -halfA, +halfA },
        origin
    );
    facesMaterialIds.push_back(0);
    
    // bottom face
    faces.emplace_back(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, +halfA },
        Point3{ -halfA, -halfA, +halfA },
        origin
    );
    facesMaterialIds.push_back(0);

    // top face
    faces.emplace_back(
        Point3{ -halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, +halfA },
        Point3{ -halfA, +halfA, +halfA },
        origin
    );
    facesMaterialIds.push_back(0);
}

// Given materials: Face
template<>
void FigureFacesConstructor::ConstructFigureFaces<FigureId::TexturedCube>
(
    std::vector<MappedSquareFace> &faces,
    std::vector<int> &facesMaterialIds,
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
    facesMaterialIds.push_back(0);

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
    facesMaterialIds.push_back(0);

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
    facesMaterialIds.push_back(0);
    
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
    facesMaterialIds.push_back(0);
    
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
    facesMaterialIds.push_back(0);

    // top face
    faces.emplace_back(
        Point3{ -halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, +halfA },
        Point3{ -halfA, +halfA, +halfA },
        origin,
        TriangleMapping{
            Point3{ 0, 0, 0 },
            Point3{ 1, 0, 0 },
            Point3{ 1, 1, 0 }
        },
        TriangleMapping{
            Point3{ 0, 0, 0 },
            Point3{ 0, 1, 0 },
            Point3{ 1, 1, 0 }
        }
    );
    facesMaterialIds.push_back(0);
}
} // namespace RayTracing
