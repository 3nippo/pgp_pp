#pragma once

#include "TriangleFace.cuh.cu"
#include "utils.cuh.cu"
#include "dummy_helper.cuh.cu"
#include "PolygonsManager.cuh.cu"


#include <vector>
#include <limits>

namespace RayTracing
{

struct BVHNode
{
aabb box;

size_t left;
size_t right;
size_t polygonIndex;

static void FromFaces(
    std::vector<MappedTriangleFace>& faces,
    std::vector<BVHNode>& nodes,
    size_t start,
    size_t end
) 
{
    nodes.emplace_back(BVHNode(faces, nodes, start, end));   
}

BVHNode(
    std::vector<MappedTriangleFace>& faces,
    std::vector<BVHNode>& nodes,
    size_t start,
    size_t end
)
{
    int axis = GenRandom(0, 2.999);

    int spanCount = end - start;
    
    polygonIndex = std::numeric_limits<size_t>::max();

    auto cmp = [axis](const MappedTriangleFace& a, const MappedTriangleFace& b)
    {
        return aabb::Compare(a.BoundingBox(), b.BoundingBox(), axis);
    };

    if (spanCount == 1)
    {
        polygonIndex = start;
        box = faces[polygonIndex].BoundingBox();

        return;
    }

    if (spanCount == 2)
    {
        nodes.emplace_back();
        nodes.emplace_back();

        if (cmp(faces[start], faces[start+1]))
        {
            left = nodes.size() - 2;
            nodes[left].polygonIndex = start;

            right = nodes.size() - 1;
            nodes[right].polygonIndex = start+1;
        }
        else
        {
            right = nodes.size() - 2;
            nodes[right].polygonIndex = start;

            left = nodes.size() - 1;
            nodes[left].polygonIndex = start+1;
        }
    }
    else
    {
        std::sort(faces.begin() + start, faces.begin() + end, cmp);

        size_t mid = start + spanCount/2;
        
        nodes.emplace_back(faces, nodes, start, mid);
        left = nodes.size() - 1;

        nodes.emplace_back(faces, nodes, mid, end);
        right = nodes.size() - 1;
    }

    box = aabb::SurroundingBox(
        nodes[left].box,
        nodes[right].box
    );
}
};

template<bool isGPU>
class BVH
{

};

template<>
class BVH<false>
{
protected:
    PolygonsManager<false> m_polygonsManager;
    std::vector<BVHNode> m_nodes;
public:
    BVH(
        std::vector<MappedTriangleFace>& faces
    )
    {
        BVHNode::FromFaces(faces, m_nodes, 0, faces.size());
    }

    void InitBeforeRender() {}
    void DeinitAfterRender() {}
    
    bool Hit(
        const Ray &ray, 
        const float tMin,
        HitRecord &hitRecord
    ) const
    {
        return HitHelper(ray, tMin, hitRecord, m_nodes.size() - 1);
    }

protected:
    bool HitHelper(
        const Ray &ray, 
        const float tMin,
        HitRecord &hitRecord,
        size_t index
    ) const
    {
        if (!m_nodes[index].box.Hit(ray, tMin, hitRecord.t))
            return false;

        if (m_nodes[index].polygonIndex != std::numeric_limits<size_t>::max())
        {
            return m_polygonsManager.Hit(
                ray,
                tMin,
                hitRecord,
                index
            );
        }

        return HitHelper(ray, tMin, hitRecord, m_nodes[index].left)
            || HitHelper(ray, tMin, hitRecord, m_nodes[index].right);
    }
};

template<>
class BVH<true> : public BVH<false>
{
private:
    PolygonsManager<true> m_polygonsManager;
    CudaMemoryLogic<BVHNode> m_nodes_d;

public:
    using BVH<false>::BVH;

    
    void InitBeforeRender() 
    {
        m_nodes_d.memcpy(this->m_nodes.data(), cudaMemcpyHostToDevice);
        this->m_nodes.clear();
    }
    void DeinitAfterRender() 
    {
        m_nodes_d.dealloc();
    }
    
    __device__
    bool Hit(
        const Ray &ray, 
        const float tMin,
        HitRecord &hitRecord
    ) const
    {
        return HitHelper(ray, tMin, hitRecord, m_nodes_d.count - 1);
    }
private:
    __device__
    bool HitHelper(
        const Ray &ray, 
        const float tMin,
        HitRecord &hitRecord,
        size_t index
    ) const
    {
        if (!m_nodes_d.get()[index].box.Hit(ray, tMin, hitRecord.t))
            return false;

        if (m_nodes_d.get()[index].polygonIndex != std::numeric_limits<size_t>::max())
        {
            return m_polygonsManager.Hit(
                ray,
                tMin,
                hitRecord,
                index
            );
        }

        return HitHelper(ray, tMin, hitRecord, m_nodes_d.get()[index].left)
            || HitHelper(ray, tMin, hitRecord, m_nodes_d.get()[index].right);
    }
};

} // namespace RayTracing
