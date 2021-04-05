#pragma once

#include <vector>

#include "SquareFace.hpp"
#include "Vector3.hpp"
#include "Ray.hpp"
#include "utils.hpp"
#include "HitRecord.hpp"
#include "Material.hpp"

#include "FigureFacesConstructor.hpp"

namespace RayTracing
{

template <FigureId figureId, typename Face>
class Figure 
{
private:
    const Point3 m_origin;
    const float m_radius;
    std::vector<Face> m_faces;
    std::vector<const Material*> m_materials;

public:
    Figure(
        const Point3 &origin, 
        const float radius
    ) 
        : m_origin(origin), m_radius(radius)
    {
        FigureFacesConstructor::ConstructFigureFaces<figureId>(
            m_faces, 
            m_materials,
            m_origin, 
            m_radius
        );
    }

    bool Hit(
        const Ray &ray, 
        const float tMin,
        HitRecord &hitRecord
    ) const
    {
        bool hitAtLeastOnce = false;

        for (size_t i = 0; i < m_faces.size(); ++i)
        {
            float tFace = 0;

            if (m_faces[i].Hit(ray, tMin, hitRecord.t, hitRecord))
            {
                hitAtLeastOnce = true;
                hitRecord.material = m_materials[i];
            }
        }

        return hitAtLeastOnce;
    }
};

using Cube = Figure<FigureId::Cube, SquareFace>;
using TexturedCube = Figure<FigureId::TexturedCube, MappedSquareFace>;
using Floor = Figure<FigureId::Floor, MappedSquareFace>;

} // namespace RayTracing
