#include <string>
#include <cmath>

#include "RayTracer.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "Figure.hpp"
#include "Vector3.hpp"
#include "Lambertian.hpp"
#include "Metallic.hpp"
#include "Image.hpp"
#include "ImageTexture.hpp"
#include "Texture.hpp"

int main()
{
    using namespace RayTracing;

    const float radius = 6;

    SolidTexture solid1(Color(0.8, 0.8, 0.0)),
                 solid2(Color(0.8, 0.8, 0.8));
    
    std::string textureFileName("morshu.data");

    Image image(textureFileName);

    ImageTexture imageTexture(image, Color(1, 1, 1));

    const Lambertian lambertian(
        &solid1
    );

    const Lambertian lambertian2(
        &imageTexture
    );

    const Metallic metallic(
        &solid2
    );

    Cube cube1(
        RayTracing::Vector3{ 0, 0, 0 },
        radius,
        &metallic
    );

    Cube cube2(
        RayTracing::Vector3{ 15, 0, 0 },
        radius,
        &lambertian2
    );

    const float radius2 = 40;

    RayTracing::Cube cube3(
        RayTracing::Vector3{ 0, (-radius2 - radius) / sqrtf(3), 0 },
        radius2,
        &lambertian
    );

    RayTracing::Scene scene(cube1, cube2, cube3);
    
    int width = 1024,
        height = 768;

    float horizontalViewDegrees = 60;

    RayTracing::Camera camera(
        width, 
        height, 
        horizontalViewDegrees,
        RayTracing::Vector3{ 6 / sqrtf(3), 0, 0 },
        RayTracing::Vector3{ 30, 5, 15 }
    );

    int sqrtSamplesPerPixel = 1;
    int depth = 4;

    RayTracing::RayTracer rayTracer(
        camera,
        scene,
        width,
        height,
        sqrtSamplesPerPixel * sqrtSamplesPerPixel,
        depth
    );

    rayTracer.Render();
    
    std::string fileName = "pic.ppm";

    rayTracer.WriteToFilePPM(fileName);

    return 0;
}
