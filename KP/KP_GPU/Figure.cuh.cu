#pragma once

#include <vector>

#include "SquareFace.cuh.cu"
#include "Vector3.cuh.cu"
#include "Ray.cuh.cu"
#include "utils.cuh.cu"
#include "HitRecord.cuh.cu"
#include "Material.cuh.cu"

#include "FigureFacesConstructor.cuh.cu"
#include "./dummy_helper.cuh.cu"

namespace RayTracing
{

template <FigureId figureId, typename Face>
class Figure 
{
private:
    CudaMemoryLogic<Face> m_faces_d;
    CudaMemoryLogic<int> m_facesMaterialIds_d;
    CudaMemoryLogic<Material**> m_materials_d;

public:
    Figure(
        const Point3 &origin, 
        const float radius,
        std::vector<Material**> materials
    ) 
    {
        std::vector<Face> faces;
        std::vector<int> facesMaterialIds;

        FigureFacesConstructor::ConstructFigureFaces<figureId>(
            faces, 
            facesMaterialIds,
            origin, 
            radius
        );

        m_faces_d.alloc(faces.size());
        m_faces_d.memcpy(faces.data(), cudaMemcpyHostToDevice);

        m_facesMaterialIds_d.alloc(facesMaterialIds.size());
        m_facesMaterialIds_d.memcpy(facesMaterialIds.data(), cudaMemcpyHostToDevice);

        m_materials_d.alloc(materials.size());
        m_materials_d.memcpy(materials.data(), cudaMemcpyHostToDevice);
    }
    
    __device__
    bool Hit(
        const Ray &ray, 
        const float tMin,
        HitRecord &hitRecord
    ) const
    {
        bool hitAtLeastOnce = false;

        for (size_t i = 0; i < m_faces_d.count; ++i)
        {
            if (m_faces_d.get()[i].Hit(ray, tMin, hitRecord.t, hitRecord))
            {
                hitAtLeastOnce = true;
                hitRecord.material = *m_materials_d.get()[m_facesMaterialIds_d.get()[i]];
            }
        }

        return hitAtLeastOnce;
    }

    void Deinit()
    {
        m_faces_d.dealloc();
        m_facesMaterialIds_d.dealloc();
        m_materials_d.dealloc();
    }
};

using Cube = Figure<FigureId::Cube, SquareFace>;
using TexturedCube = Figure<FigureId::TexturedCube, MappedSquareFace>;
using Floor = Figure<FigureId::Floor, MappedSquareFace>;
using LightSource = Floor;
using FancyCube = Figure<FigureId::FancyCube, SquareFace>;

} // namespace RayTracing
