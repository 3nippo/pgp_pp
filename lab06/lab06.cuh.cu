#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <cmath>

#include <GL/glew.h>

#include <cuda_gl_interop.h>
#include "dummy_helper.cuh.cu"

namespace lab06
{

void display();
void update();
void keys(unsigned char key, int x, int y);
void mouse(int x, int y);
void pressedMouse(int button, int state, int x, int y);

constexpr int c_gridDim  = 32;
constexpr int c_blockDim = 16;

constexpr float c_eps          = 1e-3;
constexpr float c_dt           = 1e-2;
constexpr float c_maxSpeed     = 0.05;
constexpr float c_deceleration = 0.99;
constexpr int   c_K            = 50;

class Solution
{
using TextureDataType = std::uint32_t;

public:
    Solution(
        const int w,
        const int h,
        const int np,
        const float a,
        const int n,
        const float qs,
        const float qb,
        const float qc,
        const float vb,
        const float r,
        const float speed
    );
    void GlutInit(int argc, char **argv);
    void ConfigureGL();
    void ReadSphereTexture(const std::string &textureFileName);
    void ConfigureSphereTexture();
    void ConfigureFloorTexture();
    void ConfigureFloorBuffer();
    void MainLoop();
    void InitSpheres();
    void UpdateSpheres();
    void UpdateBullet();
    void ActivateBullet();
    void UpdateCamera();
    void UpdateFloor();
    void DisplayCamera();
    void DisplaySpheres();
    void DisplayFloor();
    void DisplayCube();
    void MoveCamera(unsigned char key);
    void RotateCamera(int x, int y);
    void DisplayBullet();
    ~Solution();

public:
    struct Texture
    {
        TextureDataType w, h;
        std::vector<TextureDataType> textureBuffer;    
    };

    struct Vec2
    {
        float x;
        float y;
        Vec2(float x, float y) : x(x), y(y) {}

        Vec2& operator+=(const Vec2 &a)
        {
            x += a.x;
            y += a.y;

            return *this;
        }
    };

    struct Vec3
    {
        float x;
        float y;
        float z;
        
        Vec3(float a) : x(a), y(a), z(a) {}
        Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

        Vec3 operator*(const Vec3 &a)
        {
            Vec3 v = *this;

            v.x *= a.x;
            v.y *= a.y;
            v.z *= a.z;

            return v;
        }

        Vec3& operator*=(const Vec3 &a)
        {
            x *= a.x;
            y *= a.y;
            z *= a.z;

            return *this;
        }

        Vec3 operator/(const Vec3 &a)
        {
            Vec3 v = *this;

            v.x /= a.x;
            v.y /= a.y;
            v.z /= a.z;

            return v;
        }
        
        Vec3 operator+(const Vec3 &a)
        {
            Vec3 v = *this;

            v.x += a.x;
            v.y += a.y;
            v.z += a.z;

            return v;
        }

        Vec3& operator+=(const Vec3 &a)
        {
            x += a.x;
            y += a.y;
            z += a.z;

            return *this;
        }

        Vec3 operator-(const Vec3 &a)
        {
            Vec3 v = *this;

            v.x -= a.x;
            v.y -= a.y;
            v.z -= a.z;

            return v;
        }

        static Vec3 abs(const Vec3 &a)
        {
            return { std::abs(a.x), std::abs(a.y), std::abs(a.z) };
        }
    };

private:
    Texture m_texture;
    std::vector<Vec3> m_spheresSpeed;
    std::vector<Vec3> m_spheresPos;
    GLUquadric *m_quadratic;
    
    const int m_w;
    const int m_h;
    const int m_np;
    const float m_a;   // half of cube edge length
    const int m_n;     // number of spheres
    const float m_qs;  // spheres charge
    const float m_qb;  // bullets charge
    const float m_qc;  // cameras charge
    const float m_vb;  // bullets speed
    const float m_r;
    const float m_speed;

    Vec3 m_cameraPos{ -1.5, -1.5, 1 };
    Vec3 m_cameraSpeed{ 0, 0, 0 };

    Vec2 m_cameraAnglesPos{ 0, 0 };   // yaw, pitch
    Vec2 m_cameraAnglesSpeed{ 0, 0 }; // dyaw, dpitch

    bool m_bulletActive = false;
    Vec3 m_bulletPos{ 0, 0, 0 };
    Vec3 m_bulletSpeed{ 0, 0, 0};

    cudaGraphicsResource *m_cudaResource = nullptr;

    GLuint m_floorBufferNumber;
    GLuint m_sphereTextureNumber;
    GLuint m_floorTextureNumber;

    CudaMemory<Vec3> m_spheresPos_d;
}; // Solution

__global__
void kernel(
    uchar4 *textureData,
    int textureSide,
    Solution::Vec3 *spheresPos,
    int spheresCount,
    float a,
    float qs,
    Solution::Vec3 bulletPos,
    float qb,
    float r
);
} // lab06
