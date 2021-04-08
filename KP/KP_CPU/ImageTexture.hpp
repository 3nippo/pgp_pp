#pragma once

#include <string>

#include "Image.hpp"
#include "Texture.hpp"
#include "Vector3.hpp"

namespace RayTracing
{

class ImageTexture : public Texture
{
private:
    const Image &m_image;
    const Color m_color;

public:
    ImageTexture(
        const Image &image,
        const Color &color
    )
        : m_image(image),
          m_color(color)
    {}

private:
    virtual Color GetColor(const float u, const float v) const override
    {
        return m_color * m_image.GetColor(u, v);
    }
};

};
