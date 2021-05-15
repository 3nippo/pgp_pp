#pragma once

#include <algorithm>
#include <cmath>

#include "Vector3.cuh.cu"
#include "Ray.cuh.cu"

namespace RayTracing
{

class aabb 
{
private:
    Point3 m_min, m_max;
public:
    aabb() {}
    aabb(const Point3& a, const Point3& b) 
        : m_min(a), m_max(b)
    {}

    Point3 min() const {return m_min; }
    Point3 max() const {return m_max; }
    
    __host__ __device__
    bool Hit(const Ray& r, double t_min, double t_max) const {
        for (int a = 0; a < 3; a++) {
            auto t0 = fminf((m_min[a] - r.origin[a]) / r.direction[a],
                           (m_max[a] - r.origin[a]) / r.direction[a]);
            auto t1 = fmaxf((m_min[a] - r.origin[a]) / r.direction[a],
                           (m_max[a] - r.origin[a]) / r.direction[a]);
            t_min = fmaxf(t0, t_min);
            t_max = fminf(t1, t_max);
            if (t_max <= t_min)
                return false;
        }
        return true;
    }
};

} // RayTracing
