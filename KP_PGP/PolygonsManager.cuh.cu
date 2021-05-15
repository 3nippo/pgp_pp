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
    std::vector<std::vector<MappedTriangleFace>> m_faces;
    std::vector<aabb> m_boxes;   
public:
    PolygonsManager() {}
    
    void AddFigure(const aabb &box) 
    { 
        m_faces.emplace_back();
        m_boxes.emplace_back(box);
    }

    void AddPolygon(const TriangleFace &face)
    {
        m_faces[m_boxes.size() - 1].emplace_back(face);
    }

    void AddPolygon(const MappedTriangleFace &face)
    {
        m_faces[m_boxes.size() - 1].emplace_back(face);
    }

    void ConstructQuad(
        const Point3 &A, 
        const Point3 &B,
        const Point3 &C,
        const Point3 &D,
        const Point3 &origin,
        const Material * const * const material
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
        const Material * const * const material,
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
        HitRecord &hitRecord
    ) const
    {
        bool hitAtLeastOnce = false;

        for (int j = 0; j < m_boxes.size(); ++j)
        {
            if (!m_boxes[j].Hit(ray, tMin, hitRecord.t))
                continue;

            for (int i = 0; i < m_faces[j].size(); ++i)
                hitAtLeastOnce |= m_faces[j][i].Hit(ray, tMin, hitRecord);
        }

        return hitAtLeastOnce;
    }

    void CompleteAdding() {}

    void Deinit() {}
};

template<>
class PolygonsManager<true> : public PolygonsManager<false>
{
private:
    CudaMemoryLogic<CudaMemoryLogic<MappedTriangleFace>> m_faces_d;
    CudaMemoryLogic<aabb> m_boxes_d;
    std::vector<CudaMemoryLogic<MappedTriangleFace>> m_faces_keeper;
public:
    using PolygonsManager<false>::PolygonsManager;
    using PolygonsManager<false>::AddPolygon;
    using PolygonsManager<false>::ConstructQuad;

    __device__
    bool Hit(
        const Ray &ray, 
        const float tMin,
        HitRecord &hitRecord
    ) const
    {
        bool hitAtLeastOnce = false;

        for (int j = 0; j < m_boxes_d.count; ++j)
        {
            if (!m_boxes_d.get()[j].Hit(ray, tMin, hitRecord.t))
                continue;

            for (int i = 0; i < m_faces_d.get()[j].count; ++i)
                hitAtLeastOnce |= m_faces_d.get()[j].get()[i].Hit(ray, tMin, hitRecord);
        }

        return hitAtLeastOnce;
    }

    void CompleteAdding()
    {
        for (int i = 0; i < m_faces.size(); ++i)
        {
            m_faces_keeper.emplace_back();
            m_faces_keeper.back().alloc(this->m_faces[i].size());
            m_faces_keeper.back().memcpy(this->m_faces[i].data(), cudaMemcpyHostToDevice);
        }
        
        m_faces_d.alloc(m_faces_keeper.size());
        m_faces_d.memcpy(m_faces_keeper.data(), cudaMemcpyHostToDevice);

        m_boxes_d.alloc(this->m_boxes.size());
        m_boxes_d.memcpy(this->m_boxes.data(), cudaMemcpyHostToDevice);

        this->m_faces.clear();
        this->m_boxes.clear();
    }
    
    void Deinit()
    {
        for (int i = 0; i < m_faces_keeper.size(); ++i)
            m_faces_keeper[i].dealloc();

        m_faces_d.dealloc();
        m_boxes_d.dealloc();
    }
};

} // namespace RayTracing
