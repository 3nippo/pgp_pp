#pragma once

#include <vector>
#include <limits>

#include "Vector3.hpp"
#include "Ray.hpp"

namespace RayTracing
{

template <typename FacesConstructor, typename Face>
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
        FacesConstructor::ConstructFaces(m_faces, m_origin, m_radius);
    }

    bool Hit(
        const Ray &ray, 
        const float tMin,
        const float tMax,
        float &tOutput
    )
    {
        tOutput = std::numeric_limits<float>::infinity();

        for (size_t i = 0; i < m_faces.size(); ++i)
        {
            float tFace;

            if (m_faces[i].Hit(ray, tMin, tMax, tFace))
            {
                tOutput = std::min(tOutput, tFace);
            }
        }

        if (tOutput == std::numeric_limits<float>::infinity())
            return false;

        return true;
    }

    Vector3 GetNormalInPoint(const Point3 &point)
    {
        return (point - m_origin).UnitVector();
    }
};

} // namespace RayTracing
