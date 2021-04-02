#pragma once

#include <vector>

#include "Face.hpp"
#include "Vector3.hpp"

namespace RayTracing
{

template <typename FacesConstructor>
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
};

} // namespace RayTracing
