#include <string>

#include "RayTracer.hpp"
#include "Scene.hpp"
#include "Camera.hpp"
#include "Figure.hpp"
#include "Vector3.hpp"

int main()
{
    RayTracing::Cube cube(
        RayTracing::Vector3{ 0, 0, 0 },
        10
    );

    RayTracing::Scene scene(cube);
    
    int width = 1024,
        height = 768;

    float horizontalViewDegrees = 60;

    RayTracing::Camera camera(
        width, 
        height, 
        horizontalViewDegrees,
        RayTracing::Vector3{ 0, 0, 0 },
        RayTracing::Vector3{ 7, 7, 7 }
    );

    RayTracing::RayTracer rayTracer(
        camera,
        scene,
        width,
        height
    );
    
    std::string fileName = "pic";

    rayTracer.WriteToFile(fileName);

    return 0;
}
