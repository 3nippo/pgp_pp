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
#include "DiffuseLight.hpp"

int main()
{
    using namespace RayTracing;

    const float radius = 6;

    SolidTexture mirrorTexture(Color(0.8, 0.8, 0.8));
    
    std::string textureFileName("morshu.data");

    Image image(textureFileName);

    ImageTexture imageTexture(image, Color(1, 1, 1));

    const Lambertian lambertian2(
        &imageTexture
    );

    const Metallic metallic(
        &mirrorTexture
    );

    Cube cube1(
        RayTracing::Vector3{ 0, 0, 0 },
        radius,
        { &metallic }
    );

    TexturedCube cube2(
        RayTracing::Vector3{ 15, 0, 0 },
        radius,
        { &lambertian2 }
    );

    std::string floorFileName = "floor.data";

    Image floorImage(floorFileName);
    ImageTexture floorTexture(floorImage, Color(1, 1, 1));
    Lambertian floorMaterial(&floorTexture);
    
    float floorRadius = 10 * radius;

    Floor floor(
        Vector3{ 7.5f, -radius / sqrtf(3) - 0.001f, 0 },
        floorRadius,
        { &floorMaterial }
    );

    float lightRadius = 10;
    
    SolidTexture lightTexture(Color(8, 8, 8));
    DiffuseLight lightMaterial(&lightTexture);

    LightSource lightSource(
        { 7.5f, 4 * radius / sqrtf(3), 0 },
        lightRadius,
        { &lightMaterial }
    );

    RayTracing::Scene scene(cube1, cube2, floor, lightSource);
    
    int width = 1024,
        height = 768;

    float horizontalViewDegrees = 60;

    RayTracing::Camera camera(
        width, 
        height, 
        horizontalViewDegrees,
        RayTracing::Vector3{ 6 / sqrtf(3), 0, 0 },
        RayTracing::Vector3{ 30, 5, 15 }
        /* RayTracing::Vector3{ 60, 10, 30 } */
    );

    int sqrtSamplesPerPixel = 10;
    int depth = 5;

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
