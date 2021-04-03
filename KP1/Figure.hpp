#pragma once

#include <vector>

#include "SquareFace.hpp"
#include "Vector3.hpp"
#include "Ray.hpp"
#include "utils.hpp"
#include "HitRecord.hpp"

#include "FigureFacesConstructor.hpp"

namespace RayTracing
{

template <typename Face>
class Figure 
{
protected:
    Point3 m_origin;
    float m_radius;
    std::vector<Face> m_faces;

public:
    Figure(const Point3 &origin, const float radius) 
        : m_origin(origin), m_radius(radius)
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

        return hitAtLeastOnce;
    }
};

using Cube = Figure<SquareFace>;

} // namespace RayTracing
