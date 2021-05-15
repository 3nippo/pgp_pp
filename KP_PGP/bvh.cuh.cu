#pragma once

#include "TriangleFace.cuh.cu"
#include "utils.cuh.cu"
#include "dummy_helper.cuh.cu"
#include "PolygonsManager.cuh.cu"


#include <vector>
#include <limits>
#include <iostream>

namespace RayTracing
{
struct BVHNode
{
static constexpr size_t NULL_INDEX = std::numeric_limits<size_t>::max();
aabb box;

size_t left;
size_t right;
size_t polygonIndex;

BVHNode()
    : polygonIndex(NULL_INDEX)
{}

static void FromFaces(
    std::vector<MappedTriangleFace>& faces,
    std::vector<BVHNode>& nodes,
    size_t start,
    size_t end
) 
{
    std::vector<std::pair<MappedTriangleFace, size_t>> indexedFaces;
    
    for (size_t i = 0; i < faces.size(); ++i)
        indexedFaces.push_back({ faces[i], i });

    nodes.push_back(BVHNode(indexedFaces, nodes, start, end));   
}

BVHNode(
    std::vector<std::pair<MappedTriangleFace, size_t>>& indexedFaces,
    std::vector<BVHNode>& nodes,
    size_t start,
    size_t end
)
{
    int axis = GenRandom(0, 2.999);

    int spanCount = end - start;
    
    auto cmp = [axis](const std::pair<MappedTriangleFace, size_t>& a, const std::pair<MappedTriangleFace, size_t>& b)
    {
        return aabb::Compare(a.first.BoundingBox(), b.first.BoundingBox(), axis);
    };

    polygonIndex = NULL_INDEX;

    if (spanCount == 1)
    {
        polygonIndex = indexedFaces[start].second;
        box = indexedFaces[start].first.BoundingBox();

        return;
    }

    if (spanCount == 2)
    {
        nodes.emplace_back();
        nodes.emplace_back();

        if (cmp(indexedFaces[start], indexedFaces[start+1]))
        {
            left = nodes.size() - 2;
            nodes[left].polygonIndex = indexedFaces[start].second;
            nodes[left].box = indexedFaces[start].first.BoundingBox();

            right = nodes.size() - 1;
            nodes[right].polygonIndex = indexedFaces[start+1].second;
            nodes[right].box = indexedFaces[start+1].first.BoundingBox();
        }
        else
        {
            right = nodes.size() - 2;
            nodes[right].polygonIndex = indexedFaces[start].second;
            nodes[right].box = indexedFaces[start].first.BoundingBox();

            left = nodes.size() - 1;
            nodes[left].polygonIndex = indexedFaces[start+1].second;
            nodes[left].box = indexedFaces[start+1].first.BoundingBox();
        }
    }
    else
    {
        std::sort(indexedFaces.begin() + start, indexedFaces.begin() + end, cmp);
        size_t mid = start + spanCount/2;
        
        nodes.push_back(BVHNode(indexedFaces, nodes, start, mid));
        left = nodes.size() - 1;

        nodes.push_back(BVHNode(indexedFaces, nodes, mid, end));
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
    PolygonsManager<false> &m_polygonsManager;
    std::vector<BVHNode> m_nodes;
public:
    BVH(PolygonsManager<false>& polygonsManager)
        : m_polygonsManager(polygonsManager)
    {
        BVHNode::FromFaces(m_polygonsManager.GetFaces(), m_nodes, 0, m_polygonsManager.GetFaces().size());
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
        size_t nodeIndex,
        int depth = 0
    ) const
    {
        if (!m_nodes[nodeIndex].box.Hit(ray, tMin, hitRecord.t))
            return false;
        
        if (m_nodes[nodeIndex].polygonIndex != BVHNode::NULL_INDEX)
        {
            return m_polygonsManager.Hit(
                ray,
                tMin,
                hitRecord,
                m_nodes[nodeIndex].polygonIndex
            );
        }

        return HitHelper(ray, tMin, hitRecord, m_nodes[nodeIndex].left, depth+1) | HitHelper(ray, tMin, hitRecord, m_nodes[nodeIndex].right, depth+1);
    }
};

template<>
class BVH<true>
{
private:
    PolygonsManager<true> m_polygonsManager;
    std::vector<BVHNode> m_nodes;
    CudaMemoryLogic<BVHNode> m_nodes_d;

public:
    BVH(PolygonsManager<true>& polygonsManager)
        : m_polygonsManager(polygonsManager)
    {
        BVHNode::FromFaces(m_polygonsManager.GetFaces(), m_nodes, 0, m_polygonsManager.GetFaces().size());
    }
    
    void InitBeforeRender() 
    {
        m_nodes_d.alloc(m_nodes.size());
        m_nodes_d.memcpy(m_nodes.data(), cudaMemcpyHostToDevice);
        
        m_nodes.clear();
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
        size_t nodeIndex,
        int depth = 0
    ) const
    {
        if (!m_nodes_d.get()[nodeIndex].box.Hit(ray, tMin, hitRecord.t))
            return false;
        
        if (m_nodes_d.get()[nodeIndex].polygonIndex != BVHNode::NULL_INDEX)
        {
            return m_polygonsManager.Hit(
                ray,
                tMin,
                hitRecord,
                m_nodes_d.get()[nodeIndex].polygonIndex
            );
        }

        return HitHelper(ray, tMin, hitRecord, m_nodes_d.get()[nodeIndex].left, depth+1) | HitHelper(ray, tMin, hitRecord, m_nodes_d.get()[nodeIndex].right, depth+1);
    }
};

} // namespace RayTracing
