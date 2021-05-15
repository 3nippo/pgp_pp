#pragma once

#include <vector>

#include "Ray.cuh.cu"
#include "HitRecord.cuh.cu"
#include "dummy_helper.cuh.cu"
#include "TriangleFace.cuh.cu"
#include "Material.cuh.cu"
#include "aabb.cuh.cu"

namespace RayTracing
{

template<bool isGPU>
class PolygonsManager
{

};

template<>
class PolygonsManager<false>
{
protected:
    std::vector<MappedTriangleFace> m_faces;
public:
    PolygonsManager() {}
    
    void AddPolygon(const TriangleFace &face)
    {
        m_faces.emplace_back(face);
    }

    void AddPolygon(const MappedTriangleFace &face)
    {
        m_faces.emplace_back(face);
    }

    std::vector<MappedTriangleFace>& GetFaces()
    {
        return m_faces;
    }

    void ConstructQuad(
        const Point3 &A, 
        const Point3 &B,
        const Point3 &C,
        const Point3 &D,
        const Point3 &origin,
        Material * const * const material
    )
    {
        AddPolygon(
            TriangleFace{
                A, B, C, origin, material
            }
        );

        AddPolygon(
            TriangleFace{
                A, D, C, origin, material
            }
        );
    }

    void ConstructQuad(
        const Point3 &A, 
        const Point3 &B,
        const Point3 &C,
        const Point3 &D,
        const Point3 &origin,
        Material * const * const material,
        const TriangleMapping &mapping1,
        const TriangleMapping &mapping2
    )
    {
        AddPolygon(
            MappedTriangleFace{
                A, B, C, origin, material, mapping1
            }
        );

        AddPolygon(
            MappedTriangleFace{
                A, D, C, origin, material, mapping2
            }
        );
    }

    bool Hit(
        const Ray &ray, 
        const float tMin,
        HitRecord &hitRecord,
        size_t index
    ) const
    {
        return m_faces[index].Hit(ray, tMin, hitRecord);
    }

    void InitBeforeRender() {}

    void DeinitAfterRender() {}
};

template<>
class PolygonsManager<true> : public PolygonsManager<false>
{
private:
    CudaMemoryLogic<MappedTriangleFace> m_faces_d;
public:
    using PolygonsManager<false>::PolygonsManager;
    using PolygonsManager<false>::AddPolygon;
    using PolygonsManager<false>::ConstructQuad;
    using PolygonsManager<false>::GetFaces;

    __device__
    bool Hit(
        const Ray &ray, 
        const float tMin,
        HitRecord &hitRecord,
        size_t index
    ) const
    {
        return m_faces_d.get()[index].Hit(ray, tMin, hitRecord);
    }

    void InitBeforeRender()
    {
        m_faces_d.alloc(this->m_faces.size());
        m_faces_d.memcpy(m_faces.data(), cudaMemcpyHostToDevice);

        this->m_faces.clear();
    }
    
    void DeinitAfterRender()
    {
        m_faces_d.dealloc();
    }
};

} // namespace RayTracing
