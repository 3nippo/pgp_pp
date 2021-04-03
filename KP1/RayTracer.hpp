#pragma once

#include <vector>

#include "Camera.hpp"
#include "Vector3.hpp"
#include "Scene.hpp"

namespace RayTracing
{

class RayTracer
{
private:
    const int m_width;
    const int m_height;

    const Camera &m_camera;
    const Scene &m_scene;
    std::vector<Color> m_buffer;

public:
    RayTracer(
        const Camera &camera,
        const Scene &scene,
        const int width,
        const int height
    );
};

} // namespace RayTracing
