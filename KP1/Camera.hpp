#pragma once

#include "Vector3.hpp"
#include "Ray.hpp"

namespace RayTracing
{

class Camera
{
private:
    Point3 m_lookAt;
    Point3 m_lookFrom;

    Vector3 m_viewportHorizontal;
    Vector3 m_viewportVertical;
    
    float m_viewportHeight;
    float m_viewportWidth;

    Point3 m_lowerLeftCorner;

public:
    Camera(
        int width,
        int height,
        float horizontalViewDegrees,
        const Point3 &lookAt=Point3(),
        const Point3 &lookFrom=Point3()
    );

    void LookAt(
        const Point3 &lookAt,
        const Point3 &lookFrom
    );

    Ray GetRay(float w, float h);
};

} // namespace RayTracing
