#pragma once

#include "Figure.hpp"
#include "SquareFace.hpp"
#include "Vector3.hpp"
#include "utils.hpp"

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
        const float tMax,
        float &tOutput,
        Vector3 &normal
    ) const
    {
        tOutput = INF;
        
        float tFigure = 0;
        Vector3 normalFigure;

        if (m_cube.Hit(ray, tMin, tMax, tFigure, normalFigure) && tFigure < tOutput)
        {
            tOutput = tFigure;
            normal = normalFigure;
        }

        if (tOutput == INF)
            return false;

        return true;
    }
};

} // namespace RayTracing
