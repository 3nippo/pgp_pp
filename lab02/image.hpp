#include <string>
#include <vector>
#include <fstream>

template <class PixelType>
struct Image
{
    uint32_t width;
    uint32_t height;
    
    std::vector<PixelType> buffer;

    Image(const std::string &image_filename)
    {
        std::ifstream image_file(image_filename, std::ios_base::binary);
        
        if (!image_file.is_open())
            throw std::runtime_error("Error: couldn't open input file");
        
        image_file.read(reinterpret_cast<char*>(&width), sizeof(width));
        image_file.read(reinterpret_cast<char*>(&height), sizeof(height));
        
        while (!image_file.eof() && !image_file.fail())
        {
            PixelType pixel;

            image_file.read(reinterpret_cast<char*>(&pixel), sizeof(PixelType));

            if (image_file.eof())
                break;

            buffer.push_back(pixel);
        }

        if (image_file.bad())
            throw std::runtime_error("Error occured while reading an image");
    }

    size_t size()
    {
        return sizeof(PixelType) * width * height;
    }

    size_t count()
    {
        return width * height;
    }

    void save(const std::string& image_filename)
    {
        std::ofstream image_file(image_filename, std::ios_base::binary);

        if (!image_file.is_open())
            throw std::runtime_error("Error: couldn't open input file");

        image_file.write(reinterpret_cast<const char*>(&width), sizeof(width));
        image_file.write(reinterpret_cast<const char*>(&height), sizeof(height));
        
        for (size_t i = 0; i < buffer.size() && !image_file.fail(); ++i)
            image_file.write(reinterpret_cast<const char*>(&buffer[i]), sizeof(PixelType));
        
        if (image_file.bad())
            throw std::runtime_error("Error occured while reading an image");
    }
};
