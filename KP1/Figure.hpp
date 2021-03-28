#pragma once

#include <vector>

#include "Face.hpp"

namespace RayTracing
{

class Figure
{
protected:
    std::vector<Face> m_faces;

public:
    Figure() {}
};

} // namespace RayTracing
