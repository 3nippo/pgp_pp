#include "RayTracer.hpp"

namespace RayTracing
{

RayTracer::RayTracer(
    const Camera &camera,
    const Scene &scene,
    const int width,
    const int height
)
    : m_width(width), m_height(height), m_camera(camera), m_scene(scene)
{
    m_buffer.resize(width * height);
}

} // namespace RayTracing
