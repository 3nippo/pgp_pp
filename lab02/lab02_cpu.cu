#include <iostream>
#include <string>
#include <stdexcept>

#include "dummy_helper.cuh"
#include "image.hpp"

#define GRID_SIZE 4
#define BLOCK_SIZE 16


float to_grayscale(uchar4 pixel)
{
    float y = 0.299 * pixel.x + 0.587 * pixel.y + 0.114 * pixel.z;

    return y;
}

#define _loc(x, y, z) x[(y)*width + (z)]

void kernel(uchar4 *in, uchar4 *out, uint32_t width, uint32_t height)
{
    for (size_t y = 0; y < height; ++y)
        for (size_t x = 0; x < width; ++x)
        {
            float  y1 = to_grayscale(_loc(in,   y,   x)),
                   y2 = to_grayscale(_loc(in,   y, x+1)),
                   y3 = to_grayscale(_loc(in, y+1,   x)),
                   y4 = to_grayscale(_loc(in, y+1, x+1));
            
            unsigned char pixel_y = fminf(
                sqrtf((y1 - y4) * (y1 - y4) + (y2 - y3) * (y2 - y3)), 255
            );

            out[y * width + x] = { pixel_y, pixel_y, pixel_y, 0 };
        }
}

int submain()
{
    std::string input_name,
                output_name;

    std::cin >> input_name >> output_name;

    Image<uchar4> input_image(input_name);
    
    Image<uchar4> output_image = input_image;

    CudaTimer timer;
    
    timer.start();

    kernel(
        input_image.buffer.data(),
        output_image.buffer.data(), 
        input_image.width, 
        input_image.height
    );

    timer.stop();
    timer.print_time();

    output_image.save(output_name);

    return 0;
}

int main()
{
    try
    {
        submain();
    }
    catch (const std::exception &err)
    {
        std::cout << "ERROR:" << err.what() << std::endl;
    }

    return 0;
}
