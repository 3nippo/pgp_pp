#include "./lab04.cuh"

int main()
{
    /* try */
    /* { */
        Lab04 lab;

        lab.ReadInput();

        lab.InitGPUMemory();

        //lab.PrintCurrentState();

        lab.ForwardGaussStroke();
        
        //lab.PrintCurrentState();

        lab.BackwardGaussStroke();
        
        //lab.PrintCurrentState();

        lab.PrintX();
    /* } */
    /* catch (std::exception &err) */
    /* { */
        /* std::cout << "ERROR: " << err.what() << std::endl; */
    /* } */

    return 0;
}
