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

template <typename Face>
class Figure 
{
private:
    const Point3 m_origin;
    const float m_radius;
    std::vector<Face> m_faces;
    const Material* const m_material;

public:
    Figure(
        const Point3 &origin, 
        const float radius,
        const Material* const material
    ) 
        : m_origin(origin), m_radius(radius), m_material(material)
    {
        FigureFacesConstructor::ConstructFigureFaces(m_faces, m_origin, m_radius);
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
                hitAtLeastOnce = true;
        }

        if (hitAtLeastOnce)
            hitRecord.material = m_material;

        return hitAtLeastOnce;
    }
};

using Cube = Figure<SquareFace>;

} // namespace RayTracing
