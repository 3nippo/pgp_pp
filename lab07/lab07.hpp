#pragma once

#include <vector>
#include <string>
#include <array>

#include <mpi.h>

class Lab07
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
    int rank;

    long long block_x, block_y, block_z;

    // work data
    std::vector<double> block;

    std::vector<double> left ,
                        right,
                        front,
                        back ,
                        down ,
                        up   ;

private:
    int block_position_to_rank(
        long long block_x, 
        long long block_y, 
        long long block_z
    );

    void rank_0_init();

    void rank_non_0_init();

    void send_boundary_layers(std::vector<MPI_Request> &send_requests);

    void receive_boundary_layers(
        std::vector<double> &left,
        std::vector<double> &right,
        std::vector<double> &front,
        std::vector<double> &back,
        std::vector<double> &down,
        std::vector<double> &up,
        std::vector<MPI_Request> &receive_requests
    );

    void receive_boundary_layer(
        std::vector<double> &v, 
        int source_rank, 
        BoundaryLayerTag tag,
        std::vector<MPI_Request> &receive_requests
    );

    void send_boundary_layer(
        std::vector<double> &v, 
        int destination_rank, 
        BoundaryLayerTag tag,
        std::vector<MPI_Request> &send_requests
    );

public:
    Lab07(int argc, char **argv);

    void solve();
    void write_answer();
    static void finalize();
    static void init(int argc, char **argv);
};
