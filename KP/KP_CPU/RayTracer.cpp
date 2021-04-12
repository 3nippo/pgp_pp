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
    const int samplesPerPixel,
    const int depth
)
    : m_width(width), 
      m_height(height), 
      m_camera(camera), 
      m_scene(scene),
      m_samplesPerPixel(samplesPerPixel),
      m_depth(depth)
{
    m_buffer.resize(width * height * samplesPerPixel);
}

Color RayTracer::RayColor(const Ray &ray, const int depth)
{
    HitRecord hitRecord;

    hitRecord.t = INF;

    if (depth == 0 || !m_scene.Hit(ray, 0.001, hitRecord))
        return Color();

    Ray scattered;
    Color attenuation;
    Color computed = hitRecord.material->Emitted(hitRecord);

    if (
        hitRecord.material->Scatter(
            ray,
            hitRecord,
            attenuation,
            scattered
        )
        && hitRecord.material->reflectance != 0
    )
        computed += hitRecord.material->reflectance * attenuation * RayColor(scattered, depth - 1);

    if (hitRecord.material->transparency != 0)
        computed += \
            hitRecord.material->transparency * RayColor(
                Ray(hitRecord.point, ray.direction),
                depth - 1
            );

    return computed;
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

                m_buffer[(w + h * m_width) * m_samplesPerPixel + s] = RayColor(ray, m_depth);
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