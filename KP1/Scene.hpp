#pragma once

#include "Figure.hpp"
#include "SquareFace.hpp"
#include "Vector3.hpp"
#include "utils.hpp"

#include "HitRecord.hpp"

namespace RayTracing
{

class Scene
{
private:
    const Cube &m_cube1;
    const TexturedCube &m_cube2;
    const Floor &m_floor;
    const LightSource &m_lightSource;

public:
    Scene(
        const Cube &cube1,
        const TexturedCube &cube2,
        const Cube &cube3,
        const Floor &floor,
        const LightSource &lightSource
    )
        : m_cube1(cube1), 
          m_cube2(cube2),
          m_floor(floor),
          m_lightSource(lightSource)
    {}

    bool Hit(
        const Ray &ray, 
        const float tMin,
        HitRecord &hitRecord
    ) const
    {
        bool hitAtLeastOnce = false;

        if (m_cube1.Hit(ray, tMin, hitRecord))
            hitAtLeastOnce = true;

        if (m_cube2.Hit(ray, tMin, hitRecord))
            hitAtLeastOnce = true;

        if (m_floor.Hit(ray, tMin, hitRecord))
            hitAtLeastOnce = true;

        if (m_lightSource.Hit(ray, tMin, hitRecord))
            hitAtLeastOnce = true;

        return hitAtLeastOnce;
    }
};

} // namespace RayTracing
