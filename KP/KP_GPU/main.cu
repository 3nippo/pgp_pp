#include <string>
#include <cmath>
#include <iostream>

#include "RayTracer.cuh.cu"
#include "Scene.cuh.cu"
#include "Camera.cuh.cu"
#include "Figure.cuh.cu"
#include "Vector3.cuh.cu"
#include "Lambertian.cuh.cu"
#include "Metallic.cuh.cu"
#include "Image.cuh.cu"
#include "ImageTexture.cuh.cu"
#include "Texture.cuh.cu"
#include "DiffuseLight.cuh.cu"
#include "./DummyAllocs.cuh.cu"

#include <curand.h>
#include <curand_kernel.h>

__global__ 
void InitStates(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(42, id, 0, state + id);
}

int main()
{
try
{
    using namespace RayTracing;

    CudaMemory<curandState> states(GRID_SIZE * BLOCK_SIZE);

    InitStates<<<GRID_SIZE, BLOCK_SIZE>>>(states.get());

    const float radius = 6;
       
    CudaHeapMemory<SolidTexture, Texture, Color> mirrorTexture(Color(0, 0.749, 1)); 
    
    CudaHeapMemory<Metallic, Material, float, float, Texture**> mirrorMaterial(
        0.5,
        1,
        mirrorTexture.ptr
    );
    
    CudaHeapMemory<SolidTexture, Texture, Color> pinkTexture(Color(1, 0.07, 0.57));

    CudaHeapMemory<Lambertian, Material, float, float, Texture**, curandState*> pinkMaterial(
        0,
        1,
        pinkTexture.ptr,
        states.get()
    );
    
    CudaHeapMemory<SolidTexture, Texture, Color> edgeLightTexture(Color(2, 2, 2));

    CudaHeapMemory<DiffuseLight, Material, Texture**> edgeLightMaterial(edgeLightTexture.ptr);

    FancyCube cube1(
        Vector3{ 0, 0, 0 },
        radius,
        {
            mirrorMaterial.ptr,
            pinkMaterial.ptr,
            edgeLightMaterial.ptr
        }
    );
    
    std::string floorFileName = "floor.data";

    Image floorImage(floorFileName);

    CudaHeapMemory<ImageTexture, Texture, cudaTextureObject_t, Color> floorTexture(
        floorImage.cudaTexture,
        Color(2, 2, 2)
    );

    CudaHeapMemory<Lambertian, Material, float, float, Texture**, curandState*> floorMaterial(
        0, 
        1, 
        floorTexture.ptr,
        states.get()
    );
    /* CudaHeapMemory<Metallic, Material, float, float, Texture**> floorMaterial( */
    /*     0, */ 
    /*     1, */ 
    /*     floorTexture.ptr */
    /* ); */

    float floorRadius = 10 * radius;

    Floor floor(
        Vector3{ 7.5f, -radius / sqrtf(3) - 0.001f, 0 },
        floorRadius,
        { floorMaterial.ptr }
    );

    CudaHeapMemory<SolidTexture, Texture, Color> lightTexture(Color(8, 8, 8));

    CudaHeapMemory<DiffuseLight, Material, Texture**> lightMaterial(lightTexture.ptr);

    float lightRadius = 10;
    LightSource lightSource(
        { 2 * radius / sqrtf(3), 4 * radius / sqrtf(3), 0 },
        lightRadius,
        { lightMaterial.ptr }
    );

    RayTracing::Scene scene(cube1, floor, lightSource);
    
    int width = 1024,
        height = 768;

    float horizontalViewDegrees = 60;

    RayTracing::Camera camera(
        width, 
        height, 
        horizontalViewDegrees,
        RayTracing::Vector3(),
        RayTracing::Vector3{ 15, 0, 0 }
    );

    int sqrtSamplesPerPixel = 13;
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

    cube1.Deinit();
    floor.Deinit();
    lightSource.Deinit();

    rayTracer.WriteToFilePPM(fileName);
}
catch (std::runtime_error &err)
{
    std::cout << err.what() << std::endl;
}

    return 0;
}
