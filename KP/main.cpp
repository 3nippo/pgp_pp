#include <iostream>
#include <string>
#include <cmath>
#include <cstdlib>

#include "Config.hpp"

void PrintDefaultConfig()
{
    int err = system("cat ./default_config");
    
    if (err)
    {
        std::cout << "You probably lost config >:(" << std::endl;
    }
}

int main()
{
    Config config;

    std::cin >> config;

    return 0;
}
