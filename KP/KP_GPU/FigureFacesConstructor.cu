#include "FigureFacesConstructor.cuh.cu"
#include "SquareFace.cuh.cu"
#include <cmath>
#include "Texture.cuh.cu"
#include "utils.cuh.cu"

namespace RayTracing
{

namespace 
{

void PlaceSquaresOnEdge(
    std::vector<SquareFace> &faces,
    std::vector<int> &facesMaterialIds,
    const Point3 &start,
    const Vector3 &up,
    const Vector3 &right,
    const int n,
    const int materialIndex,
    const float a
)
{
    Vector3 shift = right.UnitVector() * a / (n+1);
    Point3 current = start + shift - right;

    for (int i = 0; i < n; ++i, current += shift)
    {
        faces.emplace_back(
            up,
            up + 2 * right,
            -up + 2 * right,
            -up,
            current
        );
        facesMaterialIds.push_back(materialIndex);
    }
}

} // namespacce

// Given materials: BigFace, SmallFace, Light
template<>
void FigureFacesConstructor::ConstructFigureFaces<FigureId::FancyCube>
(
    std::vector<SquareFace> &faces,
    std::vector<int> &facesMaterialIds,
    const Point3 &origin,
    const float radius
)
{
    constexpr int n = 6;

    float halfA = radius / sqrtf(3),
          a = halfA * 2,
          shift = halfA / n;
    
    // front face
    faces.emplace_back(
        Point3{ -halfA, -halfA, +halfA },
        Point3{ +halfA, -halfA, +halfA },
        Point3{ +halfA, +halfA, +halfA },
        Point3{ -halfA, +halfA, +halfA },
        origin + Vector3{ 0, +shift, +shift }
    );
    facesMaterialIds.push_back(0);
    
    // back face
    faces.emplace_back(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ -halfA, +halfA, -halfA },
        origin + Vector3{ 0, +shift, -shift }
    );
    facesMaterialIds.push_back(0);

    // left to back
    faces.emplace_back(
        Point3{ -halfA, -halfA, -halfA } + Vector3{ -shift, +shift, 0 },
        Point3{ -halfA, +halfA, -halfA } + Vector3{ -shift, +shift, 0 },
        Point3{ -halfA, +halfA, -halfA } + Vector3{ 0, +shift, -shift },
        Point3{ -halfA, -halfA, -halfA } + Vector3{ 0, +shift, -shift },
        origin
    );
    facesMaterialIds.push_back(1);

    // left to front
    faces.emplace_back(
        Point3{ -halfA, +halfA, +halfA } + Vector3{ -shift, +shift, 0 },
        Point3{ -halfA, -halfA, +halfA } + Vector3{ -shift, +shift, 0 },
        Point3{ -halfA, -halfA, +halfA } + Vector3{ 0, +shift, +shift },
        Point3{ -halfA, +halfA, +halfA } + Vector3{ 0, +shift, +shift },
        origin
    );
    facesMaterialIds.push_back(1);

    // left face
    faces.emplace_back(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ -halfA, +halfA, -halfA },
        Point3{ -halfA, +halfA, +halfA },
        Point3{ -halfA, -halfA, +halfA },
        origin + Vector3{ -shift, +shift, 0 }
    );
    facesMaterialIds.push_back(0);
    
    // right to back
    faces.emplace_back(
        Point3{ +halfA, -halfA, -halfA } + Vector3{ +shift, +shift, 0 },
        Point3{ +halfA, +halfA, -halfA } + Vector3{ +shift, +shift, 0 },
        Point3{ +halfA, +halfA, -halfA } + Vector3{ 0, +shift, -shift },
        Point3{ +halfA, -halfA, -halfA } + Vector3{ 0, +shift, -shift },
        origin
    );
    facesMaterialIds.push_back(1);

    // right to front
    faces.emplace_back(
        Point3{ +halfA, +halfA, +halfA } + Vector3{ +shift, +shift, 0 },
        Point3{ +halfA, -halfA, +halfA } + Vector3{ +shift, +shift, 0 },
        Point3{ +halfA, -halfA, +halfA } + Vector3{ 0, +shift, +shift },
        Point3{ +halfA, +halfA, +halfA } + Vector3{ 0, +shift, +shift },
        origin
    );
    facesMaterialIds.push_back(1);

    // right face
    faces.emplace_back(
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, +halfA },
        Point3{ +halfA, -halfA, +halfA },
        origin + Vector3{ +shift, +shift, 0 }
    );
    facesMaterialIds.push_back(0);
    
    // bottom to front
    faces.emplace_back(
        Point3{ +halfA, -halfA, +halfA },
        Point3{ -halfA, -halfA, +halfA },
        Point3{ -halfA, -halfA + shift, +halfA + shift },
        Point3{ +halfA, -halfA + shift, +halfA + shift },
        origin
    );
    facesMaterialIds.push_back(1);

    // bottom to back
    faces.emplace_back(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA + shift, -halfA - shift },
        Point3{ -halfA, -halfA + shift, -halfA - shift },
        origin
    );
    facesMaterialIds.push_back(1);

    // bottom to right
    faces.emplace_back(
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, +halfA },
        Point3{ +halfA + shift, -halfA + shift, +halfA },
        Point3{ +halfA + shift, -halfA + shift, -halfA },
        origin
    );
    facesMaterialIds.push_back(1);

    // bottom to left
    faces.emplace_back(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ -halfA, -halfA, +halfA },
        Point3{ -halfA - shift, -halfA + shift, +halfA },
        Point3{ -halfA - shift, -halfA + shift, -halfA },
        origin
    );
    facesMaterialIds.push_back(1);

    // bottom face
    faces.emplace_back(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, +halfA },
        Point3{ -halfA, -halfA, +halfA },
        origin
    );
    facesMaterialIds.push_back(0);

    // top to front
    faces.emplace_back(
        Point3{ +halfA, +halfA, +halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ -halfA, +halfA, +halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ -halfA, +halfA, +halfA } + Vector3{ 0, +shift, +shift },
        Point3{ +halfA, +halfA, +halfA } + Vector3{ 0, +shift, +shift },
        origin
    );
    facesMaterialIds.push_back(1);

    // top to back
    faces.emplace_back(
        Point3{ -halfA, +halfA, -halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ +halfA, +halfA, -halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ +halfA, +halfA, -halfA } + Vector3{ 0, +shift, -shift },
        Point3{ -halfA, +halfA, -halfA } + Vector3{ 0, +shift, -shift },
        origin
    );
    facesMaterialIds.push_back(1);

    // top to left
    faces.emplace_back(
        Point3{ -halfA, +halfA, -halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ -halfA, +halfA, +halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ -halfA, +halfA, +halfA } + Vector3{ -shift, +shift, 0 },
        Point3{ -halfA, +halfA, -halfA } + Vector3{ -shift, +shift, 0 },
        origin
    );
    facesMaterialIds.push_back(1);

    // top to right
    faces.emplace_back(
        Point3{ +halfA, +halfA, -halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ +halfA, +halfA, +halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ +halfA, +halfA, +halfA } + Vector3{ +shift, +shift, 0 },
        Point3{ +halfA, +halfA, -halfA } + Vector3{ +shift, +shift, 0 },
        origin
    );
    facesMaterialIds.push_back(1);

    // top face
    faces.emplace_back(
        Point3{ -halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, +halfA },
        Point3{ -halfA, +halfA, +halfA },
        origin + Vector3{ 0, 2*shift, 0 }
    );
    facesMaterialIds.push_back(0);
    
    const float lightHalfA = shift / 4,
                eps = 0.001;
    

    // front bottom
    PlaceSquaresOnEdge(
        faces,
        facesMaterialIds,
        origin + Vector3(-halfA, -halfA + shift / 2, -eps + halfA + shift / 2),
        Vector3{ 0, +lightHalfA, +lightHalfA },
        Vector3{ +lightHalfA, 0, 0 },
        n,
        2,
        a
    );

    // back bottom
    PlaceSquaresOnEdge(
        faces,
        facesMaterialIds,
        origin + Vector3(-halfA, -halfA + shift / 2, +eps - halfA - shift / 2),
        Vector3{ 0, +lightHalfA, -lightHalfA },
        Vector3{ +lightHalfA, 0, 0 },
        n,
        2,
        a
    );

    // left bottom
    PlaceSquaresOnEdge(
        faces,
        facesMaterialIds,
        origin + Vector3(+eps -halfA - shift / 2, -halfA + shift/2, -halfA),
        Vector3{ -lightHalfA, +lightHalfA, 0 },
        Vector3{ 0, 0, lightHalfA },
        n,
        2,
        a
    );

    // right bottom
    PlaceSquaresOnEdge(
        faces,
        facesMaterialIds,
        origin + Vector3(-eps +halfA + shift / 2, -halfA + shift/2, -halfA),
        Vector3{ +lightHalfA, +lightHalfA, 0 },
        Vector3{ 0, 0, lightHalfA },
        n,
        2,
        a
    );

    // front right
    PlaceSquaresOnEdge(
        faces,
        facesMaterialIds,
        origin + Vector3(-eps + halfA + shift/2, -halfA + shift/2, -eps + halfA + shift/2),
        Vector3{ lightHalfA, 0, -lightHalfA },
        Vector3{ 0, lightHalfA, 0},
        n,
        2,
        a
    );

    // front left
    PlaceSquaresOnEdge(
        faces,
        facesMaterialIds,
        origin + Vector3(+eps - halfA - shift/2, -halfA + shift/2, -eps + halfA + shift/2),
        Vector3{ lightHalfA, 0, lightHalfA },
        Vector3{ 0, lightHalfA, 0},
        n,
        2,
        a
    );

    // back right
    PlaceSquaresOnEdge(
        faces,
        facesMaterialIds,
        origin + Vector3(-eps + halfA + shift/2, -halfA + shift/2, +eps - halfA - shift/2),
        Vector3{ lightHalfA, 0, lightHalfA },
        Vector3{ 0, lightHalfA, 0},
        n,
        2,
        a
    );

    // back left
    PlaceSquaresOnEdge(
        faces,
        facesMaterialIds,
        origin + Vector3(+eps - halfA - shift/2, -halfA + shift/2, +eps - halfA - shift/2),
        Vector3{ lightHalfA, 0, -lightHalfA },
        Vector3{ 0, lightHalfA, 0},
        n,
        2,
        a
    );

    // top front
    PlaceSquaresOnEdge(
        faces,
        facesMaterialIds,
        origin + Vector3(-halfA, +halfA + 3 / 2.0f * shift, -eps + halfA + shift / 2),
        Vector3{ 0, +lightHalfA, -lightHalfA },
        Vector3{ +lightHalfA, 0, 0 },
        n,
        2,
        a
    );

    // top back
    PlaceSquaresOnEdge(
        faces,
        facesMaterialIds,
        origin + Vector3(-halfA, +halfA + 3 / 2.0f * shift, +eps - halfA - shift / 2),
        Vector3{ 0, +lightHalfA, +lightHalfA },
        Vector3{ +lightHalfA, 0, 0 },
        n,
        2,
        a
    );

    // top left
    PlaceSquaresOnEdge(
        faces,
        facesMaterialIds,
        origin + Vector3(+eps -halfA - shift / 2, +halfA + 3 / 2.0f * shift, -halfA),
        Vector3{ +lightHalfA, +lightHalfA, 0 },
        Vector3{ 0, 0, lightHalfA },
        n,
        2,
        a
    );

    // top right
    PlaceSquaresOnEdge(
        faces,
        facesMaterialIds,
        origin + Vector3(-eps +halfA + shift / 2, +halfA + 3 / 2.0f * shift, -halfA),
        Vector3{ -lightHalfA, +lightHalfA, 0 },
        Vector3{ 0, 0, lightHalfA },
        n,
        2,
        a
    );
}

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
