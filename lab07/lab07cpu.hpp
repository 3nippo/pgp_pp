#pragma once

#include <vector>
#include <string>
#include <array>

class Lab07cpu
{
private:
    struct Boundaries
    {
        double down;
        double up;
        double left;
        double right;
        double front;
        double back;

        bool operator==(const Boundaries &obj) const
        {
            return up == obj.up
                && down == obj.down
                && left == obj.left
                && right == obj.right
                && front == obj.front
                && back == obj.back;
        }
    };

    enum class BoundaryLayerTag
    {
        LEFT = 1, RIGHT, FRONT, BACK, DOWN, UP
    };

    static const int SEND_ANY_TAG = 0;

private:
    // input data
    std::array<long long, 3> process_grid_shape;

    std::array<long long, 3> block_shape;

    std::string output_name;

    double eps;

    std::array<double, 3> l;

    Boundaries boundaries;

    double u_0;
    
    // computed data
    std::array<long long, 3> shape;

    // work data
    std::vector<double> block;

private:
    void init();

    double& locate(long long i, long long j, long long k);
    double& locate(std::vector<double>& v, long long i, long long j, long long k);

public:
    Lab07cpu(int argc, char **argv);

    void solve();
    void write_answer();
};
