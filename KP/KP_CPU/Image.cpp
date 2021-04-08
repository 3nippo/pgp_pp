#include "Image.hpp"

#include <fstream>

#include "utils.hpp"

namespace RayTracing
{

Image::Image(const std::string &fileName)
{
    std::ifstream file(fileName);
    
    file.read(reinterpret_cast<char*>(&m_width), sizeof(m_width));
    file.read(reinterpret_cast<char*>(&m_height), sizeof(m_height));

    for (int i = 0; i < m_width * m_height; ++i)
    {
        unsigned char r = 0,
                      g = 0,
                      b = 0,
                      a = 0;

        file.read(reinterpret_cast<char*>(&r), sizeof(r));
        file.read(reinterpret_cast<char*>(&g), sizeof(g));
        file.read(reinterpret_cast<char*>(&b), sizeof(b));
        file.read(reinterpret_cast<char*>(&a), sizeof(a));

        buffer.push_back(Color{
            r / 255.0f,
            g / 255.0f,
            b / 255.0f
        });
    }
}

Color Image::GetColor(const float u, const float v) const
{
    int w = Clamp(u * (m_width - 1), 0, (m_width - 1)),
        h = Clamp(v * (m_height - 1), 0, (m_height - 1));

    return buffer[w + m_width * h];
}

} // RayTracing
