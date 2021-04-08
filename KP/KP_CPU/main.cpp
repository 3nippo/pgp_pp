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

    SolidTexture mirrorTexture(Color(0, 0.749, 1));
    
    std::string textureFileName("morshu.data");

    Image image(textureFileName);

    ImageTexture imageTexture(image, Color(1, 1, 1));

    SolidTexture pinkTexture(Color(1, 0.07, 0.57));
    Lambertian pinkMaterial(0, 1, &pinkTexture);

    Lambertian lambertian2(
        0.6,
        1,
        &imageTexture
    );

    Metallic metallic(
        0.5,
        1,
        &mirrorTexture
    );

    SolidTexture edgeLightTexture(Color(1, 1, 1));
    DiffuseLight edgeLightMaterial(&edgeLightTexture);

    FancyCube cube1(
        Vector3{ 0, 0, 0 },
        radius,
        { &metallic, &pinkMaterial, &edgeLightMaterial }
    );

    TexturedCube cube2(
        Vector3{ 15, 0, 0 },
        radius,
        { &lambertian2 }
    );

    Cube cube3(
        Vector3{ 7.5, 0, -2 * radius / sqrtf(3) },
        radius,
        { &pinkMaterial }
    );

    std::string floorFileName = "floor.data";

    Image floorImage(floorFileName);
    ImageTexture floorTexture(floorImage, Color(2, 2, 2));
    Lambertian floorMaterial(0, 1, &floorTexture);
    
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
        { 2 * radius / sqrtf(3), 4 * radius / sqrtf(3), 0 },
        lightRadius,
        { &lightMaterial }
    );

    RayTracing::Scene scene(cube1, cube2, floor, lightSource, cube3);
    
    int width = 1024,
        height = 768;

    float horizontalViewDegrees = 60;

    RayTracing::Camera camera(
        width, 
        height, 
        horizontalViewDegrees,
        RayTracing::Vector3(),
        RayTracing::Vector3{ 15, 0, 0 }
        /* RayTracing::Vector3{ 60, 10, 30 } */
    );

    int sqrtSamplesPerPixel = 4;
    int depth = 10;

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
