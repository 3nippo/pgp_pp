#pragma once

#include <vector>
#include <string>

#include "Camera.hpp"
#include "Vector3.hpp"
#include "Scene.hpp"
#include "Ray.hpp"

namespace RayTracing
{

class RayTracer
{
private:
    const int m_width;
    const int m_height;
    const int m_samplesPerPixel;

    const Camera &m_camera;
    const Scene &m_scene;
    std::vector<Color> m_buffer;

public:
    RayTracer(
        const Camera &camera,
        const Scene &scene,
        const int width,
        const int height,
        const int samplesPerPixel
    );

    void Render();

    void WriteToFile(const std::string &name);

private:
    Color RayColor(const Ray &ray);
};

} // namespace RayTracing
