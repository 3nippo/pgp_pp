#pragma once

#include <vector>
#include <string>

#include "Vector3.hpp"

namespace RayTracing
{
class Image
{
private:
    int m_width;
    int m_height;
public:
    std::vector<Color> buffer;
public:
    Image(const std::string &fileName);

    Color GetColor(const float u, const float v) const;
};
};
