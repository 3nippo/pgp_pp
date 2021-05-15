#include <algorithm>
#include <string>
#include <cmath>
#include <iostream>
#include <cstdlib>

#include <mpi.h>
#include "MPI_dummy_helper.hpp"

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

#include "bvh.cuh.cu"

#include <curand.h>
#include <curand_kernel.h>

namespace
{
int rank;

constexpr int SEND_ANY_TAG = 0;

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
    const RayTracing::Config &config,
    const int start,
    const int end
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

    // Figure B

    ObjectAllocator<SolidTexture, Texture, Color> secondMirrorTexture(config.B.color); 
    
    ObjectAllocator<Metallic, Material, float, float, Texture**> secondMirrorMaterial(
        config.B.transparency,
        config.B.reflectance,
        mirrorTexture.ptr
    );

    FigureConstructor<FigureId::FancyDodecahedron, isGPU>::ConstructFigure(
        polygonsManager,
        {
            secondMirrorMaterial.ptr,
            pinkMaterial.ptr,
            edgeLightMaterial.ptr
        },
        config.B.origin,
        config.B.radius,
        config.B.edgeLightsNum
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
    
    BVH<isGPU> bvh(polygonsManager);

    polygonsManager.InitBeforeRender();
    bvh.InitBeforeRender();
    
    RayTracer rayTracer(config, start, end);
    
    rayTracer.RenderFrames(bvh);
    
    floorImage.Deinit();
    polygonsManager.DeinitAfterRender();
    bvh.DeinitAfterRender();

    if (isGPU)
    {
        states.dealloc();
    }
}

} // namespace

void SetDevice()
{
    int device_count;

    checkCudaErrors(cudaGetDeviceCount(&device_count));

    checkCudaErrors(cudaSetDevice(rank % device_count));
}

void Finalize()
{
    int finalized;

    checkMPIErrors(MPI_Finalized(
        &finalized
    ));

    if (!finalized)
    {
        checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD));
        checkMPIErrors(MPI_Finalize());
    }    
}

int main(int argc, char **argv)
{
try
{
    int initialized;

    checkMPIErrors(MPI_Initialized(
        &initialized
    ));

    if (!initialized)
    {
        checkMPIErrors(MPI_Init(&argc, &argv));
    }

    checkMPIErrors(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    SetDevice();

    if (argc == 2 && std::string(argv[1]) == "--default")
    {
        return PrintDefaultConfig();
    }

    bool useGPU = true;

    if (argc == 2 && std::string(argv[1]) == "--cpu")
        useGPU = false;
    
    RayTracing::Config config;
    
    int start = 0, end = 0;

    if (rank == 0)
    {
        std::cin >> config;
        
        int n_ranks = 0;
        checkMPIErrors(MPI_Comm_size(MPI_COMM_WORLD, &n_ranks));

        int framesPerRank = std::max(1, config.framesNum / n_ranks);
        
        start = 0;
        end = framesPerRank;

        for (int rank = 1; rank < n_ranks; ++rank)
        {
            auto SendSome = [&rank](const void* source, const MPI_Datatype &DataType, int count)
            {
                checkMPIErrors(MPI_Send(
                    source,
                    count,
                    DataType,
                    rank,
                    SEND_ANY_TAG,
                    MPI_COMM_WORLD
                ));
            };

            int start = rank * framesPerRank,
                end = (rank+1) * framesPerRank;

            if (end > config.framesNum)
                end = -1;

            SendSome(&start, MPI_INT, 1);

            SendSome(&end, MPI_INT, 1);
            
            SendSome(&config, MPI_CHAR, sizeof(RayTracing::Config));
        }
    }
    else
    {
        auto RecvSome = [](void* dest, const MPI_Datatype &DataType, int count)
        {
            checkMPIErrors(MPI_Recv(
                dest,
                count,
                DataType,
                0,
                MPI_ANY_TAG,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            ));
        };

        RecvSome(&start, MPI_INT, 1);

        RecvSome(&end, MPI_INT, 1);
        
        RecvSome(&config, MPI_CHAR, sizeof(RayTracing::Config));

        std::cout << config.outputTemplate << std::endl;
        
        std::cout << config.width << std::endl;
        
        std::cout << config.lightSources[0].color.d.x 
                  << ' '
                  << config.lightSources[0].color.d.y
                  << ' '
                  << config.lightSources[0].color.d.z
                  << std::endl;

        std::cout << config.floorData.texturePath << std::endl;
    }
    
    if (useGPU)
        Logic<RayTracing::CudaHeapObject, true, cudaTextureObject_t>(config, start, end);
    else
        Logic<RayTracing::HeapObject, false, RayTracing::Image>(config, start, end);
}
catch (std::runtime_error &err)
{
    std::cout << err.what() << std::endl;
}

    Finalize();
    return 0;
}
