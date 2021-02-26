#include "./lab06.cuh.cu"

#include <GL/freeglut.h>
#include <iostream>
#define DEBUG() (std::cout << __LINE__ << std::endl)

constexpr int w  = 1024,
              h  = 768,
              np = 100,
              a  = 15,
              n  = 15,
              qs =  2,
              qb = 50,
              qc = 30,
              vb = 10;

constexpr float r     = 2.5,
                speed = 0.05;

lab06::Solution *solution;

void lab06::update()
{
    solution->UpdateCamera();
    solution->UpdateBullet();
    solution->UpdateSpheres();
    solution->UpdateFloor();

    glutPostRedisplay();
}

void lab06::display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(90.0f, (GLfloat)w/(GLfloat)h, 0.1f, 100.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

    solution->DisplayCamera();
    solution->DisplayBullet();
    solution->DisplaySpheres();
    solution->DisplayFloor();
    solution->DisplayCube();

	glColor3f(1.0f, 1.0f, 1.0f);

	glutSwapBuffers();
}

void lab06::keys(unsigned char key, int x, int y)
{
    if (
        key == 'w' 
        || key == 's'
        || key == 'a'
        || key == 'd'
    )
        solution->MoveCamera(key);
    else if (key == 27)
    {
        delete solution;
        exit(0);
    }
}

void lab06::pressedMouse(int button, int state, int x, int y)
{
    if (button == GLUT_LEFT_BUTTON)
    {
        solution->ActivateBullet();
    }
}

void lab06::mouse(int x, int y)
{
    solution->RotateCamera(x, y);
}

int main(int argc, char **argv)
{
    solution = new lab06::Solution(
        w, 
        h, 
        np, 
        a, 
        n, 
        qs, 
        qb, 
        qc, 
        vb, 
        r, 
        speed
    );

    solution->ReadSphereTexture("in.data");

    solution->ConfigureSphereTexture();

    solution->ConfigureFloorTexture();

    solution->ConfigureGL();

    solution->ConfigureFloorBuffer();

    solution->InitSpheres();

    solution->MainLoop();

    return 0;
}
