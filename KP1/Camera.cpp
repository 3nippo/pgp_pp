#include <cmath>

#include "Camera.hpp"
#include "utils.hpp"

namespace RayTracing
{

Camera::Camera(
    const int width,
    const int height,
    const float horizontalViewDegrees,
    const Point3 &lookAt,
    const Point3 &lookFrom
)
{
    float aspectRatio = static_cast<float>(width) / height;

    float alpha = DegreesToRadians(horizontalViewDegrees);

    m_viewportWidth = 2 * std::tan(alpha / 2); // d = 1
    m_viewportHeight = m_viewportWidth / aspectRatio;
    
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

    Vector3 w = (lookFrom - lookAt ).UnitVector(),
            u = vup.Cross(w).UnitVector(),
            v = w.Cross(u);

    m_viewportHorizontal = m_viewportWidth * u;
    m_viewportVertical = m_viewportHeight * v;
    
    m_lowerLeftCorner = m_lookFrom - m_viewportHorizontal / 2 - m_viewportVertical / 2 - w;
}

Ray Camera::GetRay(const float x, const float y) const
{
    return Ray(
        m_lookFrom,
        (m_lowerLeftCorner + x * m_viewportHorizontal + y * m_viewportVertical - m_lookFrom).UnitVector()
    );
}

} // namespace RayTracing
