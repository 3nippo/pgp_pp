#include <iostream>

#include "lab07cpu.hpp"

int main(int argc, char **argv)
{
    Lab07cpu solver(argc, argv);

    solver.solve();

    solver.write_answer();

    return 0;
}
