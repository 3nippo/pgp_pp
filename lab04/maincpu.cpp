#include "./lab04cpu.hpp"

int main()
{
    /* try */
    /* { */
        Lab04cpu lab;

        lab.ReadInput();

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
