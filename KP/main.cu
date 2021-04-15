#include <string>
#include <cmath>
#include <iostream>
#include <cstdlib>

#include "RayTracer.cuh.cu"
#include "PolygonsManager.cuh.cu"
#include "FigureConstructor.cuh.cu"
#include "Camera.cuh.cu"
#include "Vector3.cuh.cu"
#include "Lambertian.cuh.cu"
#include "Metallic.cuh.cu"
#include "Image.cuh.cu"
#include "ImageTexture.cuh.cu"
#include "Texture.cuh.cu"
#include "DiffuseLight.cuh.cu"
#include "DummyAllocs.cuh.cu"
#include "Config.cuh.cu"

#include <curand.h>
#include <curand_kernel.h>

namespace
{

__global__ 
void InitStates(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(42, id, 0, state + id);
}

int PrintDefaultConfig()
{
    int err = system("cat ./default_config");
    
    if (err)
    {
        std::cout << "You probably lost config >:(" << std::endl;
    }

    return err;
}

CudaMemoryLogic<curandState> states;

template<
    template<typename...> class ObjectAllocator,
    bool isGPU,
    typename TextureResource
>
void Logic(
    const RayTracing::Config &config
)
{
    using namespace RayTracing;

    PolygonsManager<isGPU> polygonsManager;
    
    if (isGPU)
    {
        states.alloc(GRID_SIZE * BLOCK_SIZE);

        InitStates<<<GRID_SIZE, BLOCK_SIZE>>>(states.get());
    }

    // Figure A

    ObjectAllocator<SolidTexture, Texture, Color> pinkTexture(Color(1, 0.07, 0.57));

    ObjectAllocator<Lambertian, Material, float, float, Texture**, curandState*> pinkMaterial(
        0,
        1,
        pinkTexture.ptr,
        states.get()
    );

    ObjectAllocator<SolidTexture, Texture, Color> mirrorTexture(config.A.color); 
    
    ObjectAllocator<Metallic, Material, float, float, Texture**> mirrorMaterial(
        config.A.transparency,
        config.A.reflectance,
        mirrorTexture.ptr
    );
    
    ObjectAllocator<SolidTexture, Texture, Color> edgeLightTexture(Color(2, 2, 2));

    ObjectAllocator<DiffuseLight, Material, Texture**> edgeLightMaterial(edgeLightTexture.ptr);
    
    FigureConstructor<FigureId::FancyCube, isGPU>::ConstructFigure(
        polygonsManager,
        {
            mirrorMaterial.ptr,
            pinkMaterial.ptr,
            edgeLightMaterial.ptr
        },
        config.A.origin,
        config.A.radius,
        config.A.edgeLightsNum
    );

    // LightSources

    std::vector<ObjectAllocator<
        RayTracing::SolidTexture, 
        RayTracing::Texture, 
        RayTracing::Color
    >> lightSourcesTextures;

    lightSourcesTextures.reserve(config.lightSourcesNum * 10);

    std::vector<ObjectAllocator<
        RayTracing::DiffuseLight, 
        RayTracing::Material, 
        RayTracing::Texture**
    >> lightSourcesMaterials;

    lightSourcesMaterials.reserve(config.lightSourcesNum * 10);

    for (int i = 0; i < config.lightSourcesNum; ++i)
    {
        lightSourcesTextures.emplace_back(
            config.lightSources[i].color
        );

        lightSourcesMaterials.emplace_back(
            lightSourcesTextures.back().ptr
        );

        FigureConstructor<FigureId::LightSource, isGPU>::ConstructFigure(
            polygonsManager,
            { lightSourcesMaterials[i].ptr },
            config.lightSources[i].origin,
            config.lightSources[i].radius,
            0
        );
    }

    // Floor

    Image floorImage(config.floorData.texturePath);
    
    floorImage.Init<isGPU>();

    ObjectAllocator<ImageTexture<isGPU>, Texture, TextureResource, Color> floorTexture(
        floorImage.GetResource<isGPU, TextureResource>(),
        config.floorData.color
    );

    ObjectAllocator<Lambertian, Material, float, float, Texture**, curandState*> floorMaterial(
        0, 
        config.floorData.reflectance,
        floorTexture.ptr,
        states.get()
    );

    FigureConstructor<FigureId::Floor, isGPU>::ConstructFigureByPoints(
        polygonsManager,
        { floorMaterial.ptr },
        config.floorData.A,
        config.floorData.B,
        config.floorData.C,
        config.floorData.D
    );

    polygonsManager.CompleteAdding();
    
    RayTracer rayTracer(config, 0, config.framesNum);
    
    rayTracer.RenderFrames(polygonsManager);
    
    floorImage.Deinit();
    polygonsManager.Deinit();

    if (isGPU)
    {
        states.dealloc();
    }
}

} // namespace

int main(int argc, char **argv)
{
try
{
    if (argc == 2 && std::string(argv[1]) == "--default")
    {
        return PrintDefaultConfig();
    }

    bool useGPU = true;

    if (argc == 2 && std::string(argv[1]) == "--cpu")
        useGPU = false;
    
    RayTracing::Config config;

    std::cin >> config;

    if (useGPU)
        Logic<RayTracing::CudaHeapObject, true, cudaTextureObject_t>(config);
    else
        Logic<RayTracing::HeapObject, false, RayTracing::Image>(config);
}
catch (std::runtime_error &err)
{
    std::cout << err.what() << std::endl;
}

    return 0;
}
