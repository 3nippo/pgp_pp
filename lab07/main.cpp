#include "lab07.hpp"
#include <stdexcept>
#include <iostream>

int main(int argc, char **argv)
{
    
    try
    {
        Lab07 solver(argc, argv);
        
        solver.solve();

        solver.write_answer();
    }
    catch (std::exception &err)
    {
        std::cerr << "ERROR: \n" << err.what() << std::endl;
        
        Lab07::finalize();
        return 1;
    }

    Lab07::finalize();

    return 0;
}
