#include "RayTracer.hpp"

#include <fstream>
#include <iostream>

#include "HitRecord.hpp"
#include "utils.hpp"

namespace RayTracing
{

RayTracer::RayTracer(
    const Camera &camera,
    const Scene &scene,
    const int width,
    const int height,
    const int samplesPerPixel
)
    : m_width(width), 
      m_height(height), 
      m_camera(camera), 
      m_scene(scene),
      m_samplesPerPixel(samplesPerPixel)
{
    m_buffer.resize(width * height * samplesPerPixel);
}

Color RayTracer::RayColor(const Ray &ray)
{
    HitRecord hitRecord;

    hitRecord.t = INF;

    if (m_scene.Hit(ray, 0, hitRecord))
    {
        return 0.5 * (hitRecord.normal + Color(1, 1, 1));
    }

    float s = 0.5 * (ray.direction.UnitVector().y + 1.0);

    return (1 - s) * Color(1, 1, 1) + s * Color(0.5, 0.7, 1.0);
}

void RayTracer::Render()
{
    for (int h = 0; h < m_height; ++h)
    {
        for (int w = 0; w < m_width; ++w)
        {
            for (int s = 0; s < m_samplesPerPixel; ++s)
            {
                float y = (m_height - 1 - h + GenRandom()) / (m_height - 1),
                      x = (w + GenRandom()) / (m_width - 1);

                Ray ray = m_camera.GetRay(x, y);

                m_buffer[(w + h * m_width) * m_samplesPerPixel + s] = RayColor(ray);
            }
            
            Color mean;

            for (int s = 0; s < m_samplesPerPixel; ++s)
            {
                mean += m_buffer[(w + h * m_width) * m_samplesPerPixel + s];
            }

            mean /= m_samplesPerPixel;

            mean.Clamp(0, 0.999);
            
            m_buffer[(w + h * m_width) * m_samplesPerPixel + 0] = mean;
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

    for (size_t i = 0; i < m_buffer.size(); i+=m_samplesPerPixel)
        outputFile << m_buffer[i];
}

void RayTracer::WriteToFilePPM(const std::string &name)
{
    std::ofstream outputFile(name);

    outputFile << "P3" << std::endl;

    outputFile << m_width << ' ' << m_height << std::endl;

    outputFile << 255 << std::endl;

    for (size_t i = 0; i < m_buffer.size(); i+=m_samplesPerPixel)
    {
        unsigned char r, g, b, a;

        ColorToRGBA(m_buffer[i], r, g, b, a);

        outputFile << (int)r << ' ' << (int)g << ' ' << (int)b << std::endl;
    }
}

} // namespace RayTracing
