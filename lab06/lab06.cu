#include "lab06.cuh.cu"

#include <GL/freeglut.h>
#include <cuda_runtime.h>

#include <fstream>
#include <random>
#include <iostream>
#define DEBUG() (std::cout << __LINE__ << std::endl)

lab06::Solution::Solution(
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
)
    : m_w(w),
      m_h(h),
      m_np(np),
      m_a(a),
      m_n(n),
      m_qs(qs),
      m_qb(qb),
      m_qc(qc),
      m_vb(vb),
      m_r(r),
      m_speed(speed)
{
    GlutInit(0, nullptr);
    
	m_quadratic = gluNewQuadric();
	gluQuadricTexture(m_quadratic, GL_TRUE);	

	glGenTextures(1, &m_floorTextureNumber);
	glGenTextures(1, &m_sphereTextureNumber);
    
    glewInit();   

    glGenBuffers(1, &m_floorBufferNumber);   
}

void lab06::Solution::GlutInit(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(m_w, m_h);
    glutCreateWindow("Lab06");

    glutIdleFunc(update);
    glutDisplayFunc(display);
    glutKeyboardFunc(keys);
    glutPassiveMotionFunc(mouse);
    glutMouseFunc(pressedMouse);
 
    glutSetCursor(GLUT_CURSOR_NONE);
}

void lab06::Solution::ConfigureGL()
{
    glEnable(GL_TEXTURE_2D);
    glShadeModel(GL_SMOOTH);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClearDepth(1.0f);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}

void lab06::Solution::ReadSphereTexture(const std::string &textureFileName)
{
    std::ifstream textureData(textureFileName, std::ios_base::binary);

    textureData.read(
        reinterpret_cast<char*>(&m_texture.w),
        sizeof(m_texture.w)
    );

    textureData.read(
        reinterpret_cast<char*>(&m_texture.h),
        sizeof(m_texture.h)
    );
    
    m_texture.textureBuffer
        .resize(m_texture.w * m_texture.h);

    textureData.read(
        reinterpret_cast<char*>(m_texture.textureBuffer.data()), 
        sizeof(TextureDataType) * m_texture.textureBuffer.size()
    );
}

void lab06::Solution::ConfigureSphereTexture()
{
    glBindTexture(GL_TEXTURE_2D, m_sphereTextureNumber);
    
    glTexImage2D(
        GL_TEXTURE_2D, 
        0, 
        GL_RGB, 
        (GLsizei)m_texture.w, 
        (GLsizei)m_texture.h, 
        0, 
        GL_RGBA, 
        GL_UNSIGNED_BYTE, 
        reinterpret_cast<void*>(
            m_texture.textureBuffer
                .data()
        )
    );
    m_texture.textureBuffer
        .clear();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void lab06::Solution::ConfigureFloorTexture()
{
    glBindTexture(GL_TEXTURE_2D, m_floorTextureNumber);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void lab06::Solution::ConfigureFloorBuffer()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_floorBufferNumber);
    
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_np * m_np * sizeof(uchar4), NULL, GL_DYNAMIC_DRAW);
    cudaGraphicsGLRegisterBuffer(&m_cudaResource, m_floorTextureNumber, cudaGraphicsMapFlagsWriteDiscard);
    
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void lab06::Solution::MainLoop()
{
    glutMainLoop();
}

void lab06::Solution::DisplayCamera()
{
    gluLookAt(
        m_cameraPos.x, 
        m_cameraPos.y,
        m_cameraPos.z,
        m_cameraPos.x + std::cos(m_cameraAnglesPos.x) * std::cos(m_cameraAnglesPos.y),
        m_cameraPos.y + std::sin(m_cameraAnglesPos.x) * std::cos(m_cameraAnglesPos.y),
        m_cameraPos.z + std::sin(m_cameraAnglesPos.y),
        0,
        0,
        1
    );
}

void lab06::Solution::DisplaySpheres()
{
    glBindTexture(GL_TEXTURE_2D, m_sphereTextureNumber);

    static float angle = 0.0;
	
    for (int i = 0; i < m_np; ++i)
    {
        glPushMatrix();
            glTranslatef(
                m_spheresPos[i].x,
                m_spheresPos[i].y,
                m_spheresPos[i].z
            );
            glRotatef(angle, 0.0, 0.0, 1.0);
            gluSphere(m_quadratic, m_r, 32, 32);
        glPopMatrix();
    }
	
    angle += 0.15;

    glBindTexture(GL_TEXTURE_2D, 0);
}

void lab06::Solution::DisplayFloor()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_floorBufferNumber);
	glBindTexture(GL_TEXTURE_2D, m_floorTextureNumber);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, (GLsizei)m_np, (GLsizei)m_np, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL); 
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	
	glBegin(GL_QUADS);
		glTexCoord2f(0.0, 0.0);
		glVertex3f(-m_a, -m_a, 0.0);

		glTexCoord2f(1.0, 0.0);
		glVertex3f(m_a, -m_a, 0.0);

		glTexCoord2f(1.0, 1.0);
		glVertex3f(m_a, m_a, 0.0);

		glTexCoord2f(0.0, 1.0);
		glVertex3f(-m_a, m_a, 0.0);
	glEnd();
	
	glBindTexture(GL_TEXTURE_2D, 0);
}

void lab06::Solution::DisplayCube()
{
    glLineWidth(2);
	glColor3f(0.5f, 0.5f, 0.5f);
	glBegin(GL_LINES);
		glVertex3f(-m_a, -m_a, 0.0);
		glVertex3f(-m_a, -m_a, 2.0 * m_a);

		glVertex3f(m_a, -m_a, 0.0);
		glVertex3f(m_a, -m_a, 2.0 * m_a);

		glVertex3f(m_a, m_a, 0.0);
		glVertex3f(m_a, m_a, 2.0 * m_a);

		glVertex3f(-m_a, m_a, 0.0);
		glVertex3f(-m_a, m_a, 2.0 * m_a);
	glEnd();

	glBegin(GL_LINE_LOOP);
		glVertex3f(-m_a, -m_a, 0.0);
		glVertex3f(m_a, -m_a, 0.0);
		glVertex3f(m_a, m_a, 0.0);
		glVertex3f(-m_a, m_a, 0.0);
	glEnd();

	glBegin(GL_LINE_LOOP);
		glVertex3f(-m_a, -m_a, 2.0 * m_a);
		glVertex3f(m_a, -m_a, 2.0 * m_a);
		glVertex3f(m_a, m_a, 2.0 * m_a);
		glVertex3f(-m_a, m_a, 2.0 * m_a);
	glEnd();
}

void lab06::Solution::DisplayBullet()
{
    if (!m_bulletActive)
        return;

    glPushMatrix();
        glTranslatef(
            m_bulletPos.x,
            m_bulletPos.y,
            m_bulletPos.z
        );
        gluSphere(m_quadratic, m_r / 2, 32, 32);
    glPopMatrix();
}

void lab06::Solution::MoveCamera(unsigned char key)
{
    switch (key) 
    {
		case 'w':
			m_cameraSpeed.x += cos(m_cameraAnglesPos.x) * cos(m_cameraAnglesPos.y) * m_speed;
			m_cameraSpeed.y += sin(m_cameraAnglesPos.x) * cos(m_cameraAnglesPos.y) * m_speed;
			m_cameraSpeed.z += sin(m_cameraAnglesPos.y) * m_speed;
		break;
		case 's':
			m_cameraSpeed.x += -cos(m_cameraAnglesPos.x) * cos(m_cameraAnglesPos.y) * m_speed;
			m_cameraSpeed.y += -sin(m_cameraAnglesPos.x) * cos(m_cameraAnglesPos.y) * m_speed;
			m_cameraSpeed.z += -sin(m_cameraAnglesPos.y) * m_speed;
		break;
		case 'a':
			m_cameraSpeed.x += -sin(m_cameraAnglesPos.x) * m_speed;
			m_cameraSpeed.y += cos(m_cameraAnglesPos.x) * m_speed;
			break;
		case 'd':
			m_cameraSpeed.x += sin(m_cameraAnglesPos.x) * m_speed;
			m_cameraSpeed.y += -cos(m_cameraAnglesPos.x) * m_speed;
		break;
    }
}

void lab06::Solution::RotateCamera(int x, int y)
{
    static int x_prev = m_w / 2, y_prev = m_h / 2;
	
    float dx = 0.005 * (x - x_prev);
    float dy = 0.005 * (y - y_prev);
	
    m_cameraAnglesSpeed.x -= dx;
    m_cameraAnglesSpeed.y -= dy;
	x_prev = x;
	y_prev = y;

	// Перемещаем указатель мышки в центр, когда он достиг границы
	if ((x < 20) || (y < 20) || (x > m_w - 20) || (y > m_h - 20)) {
		glutWarpPointer(m_w / 2, m_h / 2);
		x_prev = m_w / 2;
		y_prev = m_h / 2;
    }
}

void lab06::Solution::InitSpheres()
{
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<float> dist_xy(-m_a, m_a), dist_z(0, 2 * m_a);
    
    for (int i = 0; i < m_n; ++i)
    {
        m_spheresPos.emplace_back(
            dist_xy(e2), dist_xy(e2), dist_z(e2)
        );

        m_spheresSpeed.emplace_back(
            0, 0, 0
        );
    }

    m_spheresPos_d.alloc(m_n);
}

__global__
void lab06::kernel(
    uchar4 *textureData,
    int textureSide,
    lab06::Solution::Vec3 *spheresPos,
    int spheresCount,
    float a,
    float qs,
    lab06::Solution::Vec3 bulletPos,
    float qb,
    float r
)
{
    const int offsetX = blockDim.x * gridDim.x,
              offsetY = blockDim.y * gridDim.y;

    const int idX = blockIdx.x * blockDim.x + threadIdx.x,
              idY = blockIdx.y * blockDim.y + threadIdx.y;
    
#define pow2(x) ((x) * (x))
    for (int i = idX; i < textureSide; i += offsetX)
        for (int j = idY; j < textureSide; j += offsetY)
        {
            float y = (2.0 * i / (textureSide - 1.0) - 1.0) * a,
                  x = (2.0 * j / (textureSide - 1.0) - 1.0) * a;
            
            float E = 0;

            for (int p = 0; p < spheresCount; ++p)
                E += qs / (pow2(spheresPos[p].x - x) + pow2(spheresPos[p].y - y) + pow2(spheresPos[p].z - r) + c_eps);

            E += qb / (pow2(bulletPos.x - x) + pow2(bulletPos.y - y) + pow2(bulletPos.z - r / 2) + c_eps);

            E *= c_K;

            textureData[textureSide * i + j] = { 0, 0, static_cast<unsigned char>(fminf(E, 255)), 255 };
        }
#undef pow2
}

void lab06::Solution::UpdateFloor()
{
    uchar4 *textureData;
    size_t size;

	cudaGraphicsMapResources(1, &m_cudaResource, 0);
	
    cudaGraphicsResourceGetMappedPointer((void**) &textureData, &size, m_cudaResource);	// Получаем указатель на память буфера
	
    m_spheresPos_d.memcpy(m_spheresPos.data(), cudaMemcpyHostToDevice);

    kernel<<<dim3(c_gridDim, c_gridDim), dim3(c_blockDim, c_blockDim)>>>(
        textureData,
        m_np,
        m_spheresPos_d.get(),
        m_spheresPos.size(),
        m_a,
        m_qs,
        m_bulletPos,
        (m_bulletActive ? m_qb : 0),
        m_r
    );		
	
    cudaGraphicsUnmapResources(1, &m_cudaResource, 0);
}

void lab06::Solution::UpdateCamera()
{
#define norm(v) std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    float cameraSpeed = norm(m_cameraSpeed);
#undef norm

    if (cameraSpeed > c_maxSpeed)
        m_cameraSpeed *= c_maxSpeed / cameraSpeed;

    m_cameraPos += m_cameraSpeed;

    m_cameraSpeed *= c_deceleration;

    if (m_cameraPos.z < 1)
    {
        m_cameraPos.z = 1;
        m_cameraSpeed = 0;
    }

#define norm(v) (std::abs(v.x) + std::abs(v.y))
    if (norm(m_cameraAnglesSpeed) > c_eps)
    {
        m_cameraAnglesPos += m_cameraAnglesSpeed;

        m_cameraAnglesPos.y = std::min(
            static_cast<float>(M_PI / 2 - c_eps),
            std::max(
                static_cast<float>(-M_PI / 2 + c_eps),
                m_cameraAnglesPos.y
            )
        );
        
        m_cameraAnglesSpeed = { 0, 0 };
    }
#undef norm
}

void lab06::Solution::UpdateSpheres()
{
    constexpr int g = 20;
    
    std::vector< std::vector<float> > l(m_n, std::vector<float>(m_n)); // distances

#define pow2(x) ((x) * (x))
#define distance(a, b) std::sqrt(pow2(a.x - b.x) + pow2(a.y - b.y) + pow2(a.z - b.z))
    for (int i = 0; i < m_n; ++i)
        for (int j = 0; j < m_n; ++j)
        {   
            l[i][j] = distance(m_spheresPos[i], m_spheresPos[j]);

            l[j][i] = l[i][j];
        }

#define pow3(x) ((x) * (x) * (x))
    for (int i = 0; i < m_n; ++i)
    {
        m_spheresSpeed[i] *= c_deceleration;
        
        Vec3 E{ 0, 0, 0 };

        for (int j = 0; j < m_n; ++j)
        {
            if (i == j)
                continue;
            
            E += (m_spheresPos[i] - m_spheresPos[j]) * m_qs / (pow3(l[i][j]) + c_eps);
        }
        
        Vec3 wall{ -m_a, -m_a, -2*m_a };

        E += (m_spheresPos[i] + wall) * m_qs / (Vec3::abs(pow3(m_spheresPos[i] + wall)) + c_eps);
        
        wall = { m_a, m_a, 0 };
        
        E += (m_spheresPos[i] + wall) * m_qs / (Vec3::abs(pow3(m_spheresPos[i] + wall)) + c_eps);

        float l_ic = distance(m_spheresPos[i], m_cameraPos);
        
        E += (m_spheresPos[i] - m_cameraPos) * m_qc / (pow3(l_ic) + c_eps);

        if (m_bulletActive)
        {
            float l_ib = distance(m_spheresPos[i], m_bulletPos);

            E += (m_spheresPos[i] - m_bulletPos) * m_qb / (pow3(l_ib) + c_eps);
        }
        
        m_spheresSpeed[i] += E * c_K * m_qs * c_dt;

        m_spheresSpeed[i].z -= g * c_dt;
    }

    for (int i = 0; i < m_n; ++i)
        m_spheresPos[i] += m_spheresSpeed[i] * c_dt;
#define clamp(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
    for (int i = 0; i < m_n; ++i)
    {
        m_spheresPos[i].x = clamp(m_spheresPos[i].x, -m_a + m_r, m_a - m_r);
        m_spheresPos[i].y = clamp(m_spheresPos[i].y, -m_a + m_r, m_a - m_r);
        m_spheresPos[i].z = clamp(m_spheresPos[i].z, m_r, 2 * m_a - m_r);
    }
#undef clamp
#undef pow3
#undef distance
#undef pow2
}

void lab06::Solution::ActivateBullet()
{
    m_bulletActive = true;

    m_bulletSpeed.x = m_vb * std::cos(m_cameraAnglesPos.x) * std::cos(m_cameraAnglesPos.y);
    m_bulletSpeed.y = m_vb * std::sin(m_cameraAnglesPos.x) * std::cos(m_cameraAnglesPos.y);
    m_bulletSpeed.z = m_vb * std::sin(m_cameraAnglesPos.y);
    
    m_bulletPos = m_cameraPos + m_bulletSpeed;
}

void lab06::Solution::UpdateBullet()
{
    if (m_bulletActive)
    {
        m_bulletPos += m_bulletSpeed * c_dt;

        if (
            m_bulletPos.x < -2*m_a || m_bulletPos.x > 2*m_a
            || m_bulletPos.y < -2*m_a || m_bulletPos.y > 2*m_a
            || m_bulletPos.z < 0 || m_bulletPos.z > 3*m_a
        )
            m_bulletActive = false;
    }
}

lab06::Solution::~Solution()
{
    gluDeleteQuadric(m_quadratic);

    if (!!m_cudaResource)
        cudaGraphicsUnregisterResource(m_cudaResource);

    glDeleteTextures(1, &m_floorTextureNumber);
    glDeleteTextures(1, &m_floorTextureNumber);

    glDeleteBuffers(1, &m_sphereTextureNumber);
}
