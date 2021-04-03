#include "RayTracer.hpp"

#include <fstream>
#include <iostream>

#include "HitRecord.hpp"

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

Color RayTracer::RayColor(const Ray &ray)
{
    HitRecord hitRecord;

    hitRecord.t = INF;

    if (m_scene.Hit(ray, 0, hitRecord))
    {
        return 0.5 * (hitRecord.normal + Color(1, 1, 1));
        /* return Color(0, 0, 0); */
    }

    float s = 0.5 * (ray.direction.UnitVector().y + 1.0);

    return (1 - s) * Color(1, 1, 1) + s * Color(0.5, 0.7, 1.0);
}

void RayTracer::Render()
{
    /* Ray ray = m_camera.GetRay(0.5, 0.5); */
    
    /* Color color = RayColor(ray); */

    /* std::cout << color.x << ' ' << color.y << ' ' << color.z << std::endl; */

    for (int h = 0; h < m_height; ++h)
    {
        float y = static_cast<float>(m_height - 1 - h) / (m_height - 1);

        for (int w = 0; w < m_width; ++w)
        {
            float x = static_cast<float>(w) / (m_width - 1);

            Ray ray = m_camera.GetRay(x, y);

            m_buffer[w + h * m_width] = RayColor(ray);
        }
    }
}

void RayTracer::WriteToFile(const std::string &name)
{
    std::ofstream outputFile(name);

    outputFile.write(
        reinterpret_cast<const char*>(&m_width), 
        sizeof(m_width)
    );

    outputFile.write(
        reinterpret_cast<const char*>(&m_height),
        sizeof(m_height)
    );

    for (size_t i = 0; i < m_buffer.size(); ++i)
        outputFile << m_buffer[i];
}

} // namespace RayTracing
