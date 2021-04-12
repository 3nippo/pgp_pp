#pragma once

#include "Figure.cuh.cu"
#include "SquareFace.cuh.cu"
#include "Vector3.cuh.cu"
#include "utils.cuh.cu"

#include "HitRecord.cuh.cu"

namespace RayTracing
{

class Scene
{
private:
    FancyCube m_cube1;
    /* const TexturedCube m_cube2; */
    Floor m_floor;
    LightSource m_lightSource;
    /* const Cube m_cube3; */

public:
    Scene(
        FancyCube&cube1,
        /* const TexturedCube &cube2, */
        Floor &floor,
        LightSource &lightSource
        /* const Cube &cube3 */
    )
        : m_cube1(cube1), 
          /* m_cube2(cube2), */
          m_floor(floor),
          m_lightSource(lightSource)
          /* m_cube3(cube3) */
    {
    }
    
    __device__
    bool Hit(
        const Ray &ray, 
        const float tMin,
        HitRecord &hitRecord
    ) const
    {
        bool hitAtLeastOnce = false;

        if (m_cube1.Hit(ray, tMin, hitRecord))
            hitAtLeastOnce = true;

        /* if (m_cube2.Hit(ray, tMin, hitRecord)) */
        /*     hitAtLeastOnce = true; */

        if (m_floor.Hit(ray, tMin, hitRecord))
            hitAtLeastOnce = true;

        if (m_lightSource.Hit(ray, tMin, hitRecord))
            hitAtLeastOnce = true;

        /* if (m_cube3.Hit(ray, tMin, hitRecord)) */
        /*     hitAtLeastOnce = true; */


        return hitAtLeastOnce;
    }
};

} // namespace RayTracing