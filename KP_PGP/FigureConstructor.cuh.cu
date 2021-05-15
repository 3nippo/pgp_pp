#pragma once

#include <vector>
#include <cassert>
#include <fstream>
#include <string>

#include "Vector3.cuh.cu"
#include "PolygonsManager.cuh.cu"
#include "Material.cuh.cu"
#include "TriangleFace.cuh.cu"


namespace RayTracing
{

namespace 
{

template<bool isGPU>
void PlaceSquaresOnEdge(
    PolygonsManager<isGPU> &polygonsManager,
    Material * const * const material,
    const Point3 &start,
    const Vector3 &up,
    const Vector3 &right,
    const int n,
    const float a
)
{
    Vector3 shift = right.UnitVector() * a / (n+1);
    Point3 current = start + shift - right;

    /* shift = shift / a * (a + shift.Length() / 2); */

    for (int i = 0; i < n; ++i, current += shift)
    {
        polygonsManager.ConstructQuad(
            up,
            up + 2 * right,
            -up + 2 * right,
            -up,
            current,
            material
        );
    }
}

} // namespacce

enum class FigureId
{
    Cube,
    TexturedCube,
    Floor,
    FancyCube,
    FancyDodecahedron,
    LightSource=Floor
};

template<FigureId figureId, bool isGPU>
struct FigureConstructor
{
    static void ConstructFigure(
        PolygonsManager<isGPU> &polygonsManager,
        const std::vector<Material**> &materials,
        const Point3 &origin,
        const float radius,
        const int edgeLightsNum
    )
    {
        assert(("Not implemented", false));
    }

    static void ConstructFigureByPoints(
        PolygonsManager<isGPU> &polygonsManager,
        const std::vector<Material**> &materials,
        const Vector3 &A,
        const Vector3 &B,
        const Vector3 &C,
        const Vector3 &D
    )
    {
        assert(("Not implemented", false));
    }
};

template<bool isGPU>
void BuildPolygonsFromFile(
    const std::string &filename,
    PolygonsManager<isGPU> &polygonsManager,
    Material * const * const material,
    const Point3 &origin,
    const float radius
)
{
    std::ifstream obj(filename);
    
    std::vector<Vector3> vertices;

    std::string token;
    
    while (obj >> token)
    {
        if (token == "v")
        {
            vertices.emplace_back();

            obj >> vertices.back();

            vertices.back() = vertices.back() * radius;
        }
        else if (token == "f")
        {
            std::string delimiter;

            int indices[3];
            int ignore;

            obj >> indices[0] 
                >> delimiter
                >> ignore
                >> indices[1]
                >> delimiter
                >> ignore
                >> indices[2]
                >> delimiter
                >> ignore;
            
            polygonsManager.AddPolygon(TriangleFace(
                vertices[indices[0] - 1],
                vertices[indices[1] - 1],
                vertices[indices[2] - 1],
                origin,
                material
            ));
        }
        else
        {
            std::cout << token << std::endl;
            assert(("Something went wrong", false));
        }
    }
}

template<bool isGPU>
void BuildLightsFromFile(
    const std::string &filename,
    PolygonsManager<isGPU> &polygonsManager,
    Material * const * const material,
    const Point3 &origin,
    const float radius,
    const int lightsNum
)
{
    std::ifstream obj(filename);
    
    std::vector<Vector3> vertices,
                         normals;

    std::string token;

    while (obj >> token)
    {
        if (token == "v")
        {
            vertices.emplace_back();

            obj >> vertices.back();

            vertices.back() = vertices.back() * radius + origin;
        }
        else if (token == "vn")
        {
            normals.emplace_back();

            obj >> normals.back();
        }
        else if (token == "f")
        {
            std::string delimiter;

            int indices[4];
            int normalIndex;

            obj >> indices[0] 
                >> delimiter
                >> normalIndex
                >> indices[1]
                >> delimiter
                >> normalIndex
                >> indices[2]
                >> delimiter
                >> normalIndex
                >> indices[3]
                >> delimiter
                >> normalIndex;
            
            Vector3 a = vertices[indices[0] - 1] - vertices[indices[1] - 1];
            Vector3 b = vertices[indices[2] - 1] - vertices[indices[1] - 1];

            if (a.Length() > b.Length())
                std::swap(a, b);

            constexpr float eps = 0.001f;
            Vector3 lightOrigin = vertices[indices[1] - 1] + a / 2;
            
            float side = b.Length();

            a = a * 0.3 / 2;
            b = b.UnitVector() * a.Length();

            PlaceSquaresOnEdge(
                polygonsManager,
                material,
                lightOrigin - eps * normals[normalIndex - 1].UnitVector(),
                a,
                b,
                lightsNum,
                side
            );
        }
        else
        {
            std::cout << token << std::endl;
            assert(("Something went wrong", false));
        }
    }
}

// Given materials: BigFace, SmallFace, Light
template<bool isGPU>
struct FigureConstructor<FigureId::FancyDodecahedron, isGPU>
{
static void ConstructFigure(
    PolygonsManager<isGPU> &polygonsManager,
    const std::vector<Material**> &materials,
    const Point3 &origin,
    const float radius,
    const int edgeLightsNum
)
{
    // big fases

    BuildPolygonsFromFile(
        "./penta.obj",
        polygonsManager,
        materials[0],
        origin,
        radius
    );

    // small faces
    
    BuildPolygonsFromFile(
        "./polyedges.obj",
        polygonsManager,
        materials[1],
        origin,
        radius
    );

    // lights

    BuildLightsFromFile(
        "./edges.obj",
        polygonsManager,
        materials[2],
        origin,
        radius,
        edgeLightsNum
    );
}
}; // FancyDodecahedron

// Given materials: BigFace, SmallFace, Light
template<bool isGPU>
struct FigureConstructor<FigureId::FancyCube, isGPU>
{
static void ConstructFigure(
    PolygonsManager<isGPU> &polygonsManager,
    const std::vector<Material**> &materials,
    const Point3 &origin,
    const float radius,
    const int edgeLightsNum
)
{
    float halfA = radius / sqrtf(3),
          a = halfA * 2,
          shift = halfA / edgeLightsNum;
    
    // front face
    polygonsManager.ConstructQuad(
        Point3{ -halfA, -halfA, +halfA },
        Point3{ +halfA, -halfA, +halfA },
        Point3{ +halfA, +halfA, +halfA },
        Point3{ -halfA, +halfA, +halfA },
        origin + Vector3{ 0, +shift, +shift },
        materials[0]
    );
    
    // back face
    polygonsManager.ConstructQuad(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ -halfA, +halfA, -halfA },
        origin + Vector3{ 0, +shift, -shift },
        materials[0]
    );

    // left to back
    polygonsManager.ConstructQuad(
        Point3{ -halfA, -halfA, -halfA } + Vector3{ -shift, +shift, 0 },
        Point3{ -halfA, +halfA, -halfA } + Vector3{ -shift, +shift, 0 },
        Point3{ -halfA, +halfA, -halfA } + Vector3{ 0, +shift, -shift },
        Point3{ -halfA, -halfA, -halfA } + Vector3{ 0, +shift, -shift },
        origin,
        materials[1]
    );

    // left to front
    polygonsManager.ConstructQuad(
        Point3{ -halfA, +halfA, +halfA } + Vector3{ -shift, +shift, 0 },
        Point3{ -halfA, -halfA, +halfA } + Vector3{ -shift, +shift, 0 },
        Point3{ -halfA, -halfA, +halfA } + Vector3{ 0, +shift, +shift },
        Point3{ -halfA, +halfA, +halfA } + Vector3{ 0, +shift, +shift },
        origin,
        materials[1]
    );

    // left face
    polygonsManager.ConstructQuad(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ -halfA, +halfA, -halfA },
        Point3{ -halfA, +halfA, +halfA },
        Point3{ -halfA, -halfA, +halfA },
        origin + Vector3{ -shift, +shift, 0 },
        materials[0]
    );
    
    // right to back
    polygonsManager.ConstructQuad(
        Point3{ +halfA, -halfA, -halfA } + Vector3{ +shift, +shift, 0 },
        Point3{ +halfA, +halfA, -halfA } + Vector3{ +shift, +shift, 0 },
        Point3{ +halfA, +halfA, -halfA } + Vector3{ 0, +shift, -shift },
        Point3{ +halfA, -halfA, -halfA } + Vector3{ 0, +shift, -shift },
        origin,
        materials[1]
    );

    // right to front
    polygonsManager.ConstructQuad(
        Point3{ +halfA, +halfA, +halfA } + Vector3{ +shift, +shift, 0 },
        Point3{ +halfA, -halfA, +halfA } + Vector3{ +shift, +shift, 0 },
        Point3{ +halfA, -halfA, +halfA } + Vector3{ 0, +shift, +shift },
        Point3{ +halfA, +halfA, +halfA } + Vector3{ 0, +shift, +shift },
        origin,
        materials[1]
    );

    // right face
    polygonsManager.ConstructQuad(
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, +halfA },
        Point3{ +halfA, -halfA, +halfA },
        origin + Vector3{ +shift, +shift, 0 },
        materials[0]
    );
    
    // bottom to front
    polygonsManager.ConstructQuad(
        Point3{ +halfA, -halfA, +halfA },
        Point3{ -halfA, -halfA, +halfA },
        Point3{ -halfA, -halfA + shift, +halfA + shift },
        Point3{ +halfA, -halfA + shift, +halfA + shift },
        origin,
        materials[1]
    );

    // bottom to back
    polygonsManager.ConstructQuad(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA + shift, -halfA - shift },
        Point3{ -halfA, -halfA + shift, -halfA - shift },
        origin,
        materials[1]
    );

    // bottom to right
    polygonsManager.ConstructQuad(
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, +halfA },
        Point3{ +halfA + shift, -halfA + shift, +halfA },
        Point3{ +halfA + shift, -halfA + shift, -halfA },
        origin,
        materials[1]
    );

    // bottom to left
    polygonsManager.ConstructQuad(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ -halfA, -halfA, +halfA },
        Point3{ -halfA - shift, -halfA + shift, +halfA },
        Point3{ -halfA - shift, -halfA + shift, -halfA },
        origin,
        materials[1]
    );

    // bottom face
    polygonsManager.ConstructQuad(
        Point3{ -halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, -halfA },
        Point3{ +halfA, -halfA, +halfA },
        Point3{ -halfA, -halfA, +halfA },
        origin,
        materials[0]
    );

    // top to front
    polygonsManager.ConstructQuad(
        Point3{ +halfA, +halfA, +halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ -halfA, +halfA, +halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ -halfA, +halfA, +halfA } + Vector3{ 0, +shift, +shift },
        Point3{ +halfA, +halfA, +halfA } + Vector3{ 0, +shift, +shift },
        origin,
        materials[1]
    );

    // top to back
    polygonsManager.ConstructQuad(
        Point3{ -halfA, +halfA, -halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ +halfA, +halfA, -halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ +halfA, +halfA, -halfA } + Vector3{ 0, +shift, -shift },
        Point3{ -halfA, +halfA, -halfA } + Vector3{ 0, +shift, -shift },
        origin,
        materials[1]
    );

    // top to left
    polygonsManager.ConstructQuad(
        Point3{ -halfA, +halfA, -halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ -halfA, +halfA, +halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ -halfA, +halfA, +halfA } + Vector3{ -shift, +shift, 0 },
        Point3{ -halfA, +halfA, -halfA } + Vector3{ -shift, +shift, 0 },
        origin,
        materials[1]
    );

    // top to right
    polygonsManager.ConstructQuad(
        Point3{ +halfA, +halfA, -halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ +halfA, +halfA, +halfA } + Vector3{ 0, 2*shift, 0 },
        Point3{ +halfA, +halfA, +halfA } + Vector3{ +shift, +shift, 0 },
        Point3{ +halfA, +halfA, -halfA } + Vector3{ +shift, +shift, 0 },
        origin,
        materials[1]
    );

    // top face
    polygonsManager.ConstructQuad(
        Point3{ -halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, -halfA },
        Point3{ +halfA, +halfA, +halfA },
        Point3{ -halfA, +halfA, +halfA },
        origin + Vector3{ 0, 2*shift, 0 },
        materials[0]
    );
    
    const float lightHalfA = shift / 4,
                eps = 0.001;

    // front bottom
    PlaceSquaresOnEdge(
        polygonsManager,
        materials[2],
        origin + Vector3(-halfA, -halfA + shift / 2, -eps + halfA + shift / 2),
        Vector3{ 0, +lightHalfA, +lightHalfA },
        Vector3{ +lightHalfA, 0, 0 },
        edgeLightsNum,
        a
    );

    // back bottom
    PlaceSquaresOnEdge(
        polygonsManager,
        materials[2],
        origin + Vector3(-halfA, -halfA + shift / 2, +eps - halfA - shift / 2),
        Vector3{ 0, +lightHalfA, -lightHalfA },
        Vector3{ +lightHalfA, 0, 0 },
        edgeLightsNum,
        a
    );

    // left bottom
    PlaceSquaresOnEdge(
        polygonsManager,
        materials[2],
        origin + Vector3(+eps -halfA - shift / 2, -halfA + shift/2, -halfA),
        Vector3{ -lightHalfA, +lightHalfA, 0 },
        Vector3{ 0, 0, lightHalfA },
        edgeLightsNum,
        a
    );

    // right bottom
    PlaceSquaresOnEdge(
        polygonsManager,
        materials[2],
        origin + Vector3(-eps +halfA + shift / 2, -halfA + shift/2, -halfA),
        Vector3{ +lightHalfA, +lightHalfA, 0 },
        Vector3{ 0, 0, lightHalfA },
        edgeLightsNum,
        a
    );

    // front right
    PlaceSquaresOnEdge(
        polygonsManager,
        materials[2],
        origin + Vector3(-eps + halfA + shift/2, -halfA + shift/2, -eps + halfA + shift/2),
        Vector3{ lightHalfA, 0, -lightHalfA },
        Vector3{ 0, lightHalfA, 0},
        edgeLightsNum,
        a
    );

    // front left
    PlaceSquaresOnEdge(
        polygonsManager,
        materials[2],
        origin + Vector3(+eps - halfA - shift/2, -halfA + shift/2, -eps + halfA + shift/2),
        Vector3{ lightHalfA, 0, lightHalfA },
        Vector3{ 0, lightHalfA, 0},
        edgeLightsNum,
        a
    );

    // back right
    PlaceSquaresOnEdge(
        polygonsManager,
        materials[2],
        origin + Vector3(-eps + halfA + shift/2, -halfA + shift/2, +eps - halfA - shift/2),
        Vector3{ lightHalfA, 0, lightHalfA },
        Vector3{ 0, lightHalfA, 0},
        edgeLightsNum,
        a
    );

    // back left
    PlaceSquaresOnEdge(
        polygonsManager,
        materials[2],
        origin + Vector3(+eps - halfA - shift/2, -halfA + shift/2, +eps - halfA - shift/2),
        Vector3{ lightHalfA, 0, -lightHalfA },
        Vector3{ 0, lightHalfA, 0},
        edgeLightsNum,
        a
    );

    // top front
    PlaceSquaresOnEdge(
        polygonsManager,
        materials[2],
        origin + Vector3(-halfA, +halfA + 3 / 2.0f * shift, -eps + halfA + shift / 2),
        Vector3{ 0, +lightHalfA, -lightHalfA },
        Vector3{ +lightHalfA, 0, 0 },
        edgeLightsNum,
        a
    );

    // top back
    PlaceSquaresOnEdge(
        polygonsManager,
        materials[2],
        origin + Vector3(-halfA, +halfA + 3 / 2.0f * shift, +eps - halfA - shift / 2),
        Vector3{ 0, +lightHalfA, +lightHalfA },
        Vector3{ +lightHalfA, 0, 0 },
        edgeLightsNum,
        a
    );

    // top left
    PlaceSquaresOnEdge(
        polygonsManager,
        materials[2],
        origin + Vector3(+eps -halfA - shift / 2, +halfA + 3 / 2.0f * shift, -halfA),
        Vector3{ +lightHalfA, +lightHalfA, 0 },
        Vector3{ 0, 0, lightHalfA },
        edgeLightsNum,
        a
    );

    // top right
    PlaceSquaresOnEdge(
        polygonsManager,
        materials[2],
        origin + Vector3(-eps +halfA + shift / 2, +halfA + 3 / 2.0f * shift, -halfA),
        Vector3{ -lightHalfA, +lightHalfA, 0 },
        Vector3{ 0, 0, lightHalfA },
        edgeLightsNum,
        a
    );
}
}; // FancyCube

// Given materials: Floor
template<bool isGPU>
struct FigureConstructor<FigureId::Floor, isGPU>
{
static void ConstructFigure(
    PolygonsManager<isGPU> &polygonsManager,
    const std::vector<Material**> &materials,
    const Point3 &origin,
    const float radius,
    const int edgeLightsNum
)
{
    float halfA = radius / sqrtf(3);

    polygonsManager.ConstructQuad(
        Point3{ -halfA, 0, -halfA },
        Point3{ +halfA, 0, -halfA },
        Point3{ +halfA, 0, +halfA },
        Point3{ -halfA, 0, +halfA },
        origin,
        materials[0],
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
}

static void ConstructFigureByPoints(
    PolygonsManager<isGPU> &polygonsManager,
    const std::vector<Material**> &materials,
    const Vector3 &A,
    const Vector3 &B,
    const Vector3 &C,
    const Vector3 &D
)
{
    polygonsManager.ConstructQuad(
        A,
        B,
        C,
        D,
        { 0, 0, 0, 0 },
        materials[0],
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
}
}; // Floor

} // namespace RayTracing
