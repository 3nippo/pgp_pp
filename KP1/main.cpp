#include <string>

#include "RayTracer.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "Figure.hpp"
#include "Vector3.hpp"

int main()
{
    RayTracing::Cube cube1(
        RayTracing::Vector3{ 0, 0, 0 },
        6
    );

    RayTracing::Cube cube2(
        RayTracing::Vector3{ 15, 0, 0 },
        6
    );

    RayTracing::Scene scene(cube1, cube2);
    
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

    RayTracing::RayTracer rayTracer(
        camera,
        scene,
        width,
        height
    );

    rayTracer.Render();
    
    std::string fileName = "pic";

    rayTracer.WriteToFile(fileName);

    return 0;
}
