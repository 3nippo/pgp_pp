#include <cmath>

#include "Camera.hpp"
#include "utils.hpp"

namespace RayTracing
{

Camera::Camera(
    int width,
    int height,
    float horizontalViewDegrees,
    const Point3 &lookAt,
    const Point3 &lookFrom
)
{
    float aspectRatio = static_cast<float>(width) / height;

    float alpha = DegreesToRadiand(horizontalViewDegrees);

    m_viewportHeight = 2 * std::tan(alpha/2), // d = 1
    m_viewportWidth = aspectRatio * m_viewportHeight;
    
    LookAt(lookAt, lookFrom);
}

void Camera::LookAt(
    const Point3 &lookAt,
    const Point3 &lookFrom
)
{
    m_lookFrom = lookFrom;
    m_lookAt = lookAt;

    const Vector3 vup{ 0, 1, 0 };

    Vector3 w = (lookAt - lookFrom).UnitVector(),
            u = vup.Cross(w).UnitVector(),
            v = w.Cross(u);

    m_viewportHorizontal = m_viewportHeight * u;
    m_viewportVertical = m_viewportWidth * v;
    
    m_lowerLeftCorner = m_lookFrom - m_viewportHorizontal / 2 - m_viewportVertical / 2 - w;
}

Ray Camera::GetRay(float x, float y)
{
    return Ray(
        m_lookFrom,
        m_lowerLeftCorner + x * m_viewportHorizontal + y * m_viewportVertical - m_lookFrom
    );
}

} // namespace RayTracing
