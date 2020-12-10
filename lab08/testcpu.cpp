#include <iostream>

#include "lab08cpu.hpp"

int main(int argc, char **argv)
{
    Lab08cpu solver(argc, argv);

    solver.solve();

    solver.write_answer();

    return 0;
}
