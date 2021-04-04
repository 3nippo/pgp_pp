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

    std::vector<Color> m_buffer;
public:
    Image(const std::string &fileName);

    Color GetColor(const float u, const float v) const;
};
};
