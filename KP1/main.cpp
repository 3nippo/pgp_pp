#include <string>
#include <cmath>

#include "RayTracer.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "Figure.hpp"
#include "Vector3.hpp"

int main()
{
    const float radius = 6;

    RayTracing::Cube cube1(
        RayTracing::Vector3{ 0, 0, 0 },
        radius
    );

    RayTracing::Cube cube2(
        RayTracing::Vector3{ 15, 0, 0 },
        radius
    );

    const float radius2 = 40;

    RayTracing::Cube cube3(
        RayTracing::Vector3{ 0, (-radius2 - radius) / sqrtf(3), 0 },
        radius2
    );

    RayTracing::Scene scene(cube1, cube2, cube3);
    
    int width = 1024,
        height = 768;

    float horizontalViewDegrees = 60;

    RayTracing::Camera camera(
        width, 
        height, 
        horizontalViewDegrees,
        RayTracing::Vector3{ 7.5, 0, 0 },
        RayTracing::Vector3{ 17, 17, 17 }
    );

    int sqrtSamplesPerPixel = 4;

    RayTracing::RayTracer rayTracer(
        camera,
        scene,
        width,
        height,
        sqrtSamplesPerPixel * sqrtSamplesPerPixel
    );

    rayTracer.Render();
    
    std::string fileName = "pic.ppm";

    rayTracer.WriteToFilePPM(fileName);

    return 0;
}
