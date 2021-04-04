#include <string>
#include <cmath>

#include "RayTracer.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "Figure.hpp"
#include "Vector3.hpp"
#include "Lambertian.hpp"

int main()
{
    const float radius = 6;

    const RayTracing::Lambertian lambertian(
        RayTracing::Color(1, 1, 1)
    );

    RayTracing::Cube cube1(
        RayTracing::Vector3{ 0, 0, 0 },
        radius,
        &lambertian
    );

    RayTracing::Cube cube2(
        RayTracing::Vector3{ 15, 0, 0 },
        radius,
        &lambertian
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
        RayTracing::Vector3{ 7.5, 0, 0 },
        RayTracing::Vector3{ 17, 17, 17 }
    );

    int sqrtSamplesPerPixel = 4;
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
