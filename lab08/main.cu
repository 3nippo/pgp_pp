#include "lab08.cuh"
#include <stdexcept>
#include <iostream>

int main(int argc, char **argv)
{
    
    try
    {
        Lab08 solver(argc, argv);
        
        solver.solve();

        solver.write_answer();
    }
    catch (std::exception &err)
    {
        std::cerr << "ERROR: \n" << err.what() << std::endl;
        
        Lab08::finalize();
        return 1;
    }

    Lab08::finalize();

    return 0;
}
