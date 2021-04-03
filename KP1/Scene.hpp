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
    Figure<SquareFace> m_cube;

public:
    Scene(
        const Figure<SquareFace> &cube
    )
        : m_cube(cube)
    {}

    bool Hit(
        const Ray &ray, 
        const float tMin,
        HitRecord &hitRecord
    ) const
    {
        if (m_cube.Hit(ray, tMin, hitRecord))
            return true;

        return false;
    }
};

} // namespace RayTracing
