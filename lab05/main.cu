#include <stdexcept>
#include <iostream>

#include "./lab05.cuh"

int main()
{
    try
    {
        Lab05 lab;

        lab.ReadInput();
        
        lab.InitGPUMemory();

        lab.Sort();

        lab.PrintTextResult();

        //lab.PrintResult();
    }
    catch (const std::exception &err)
    {
        std::cout << "ERROR: " << err.what() << std::endl;
    }
}
