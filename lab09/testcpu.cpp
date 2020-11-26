#include <iostream>

#include "lab09cpu.hpp"

int main(int argc, char **argv)
{
    Lab09cpu solver(argc, argv);

    solver.solve();

    solver.write_answer();

    return 0;
}
