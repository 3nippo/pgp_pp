#pragma once

#include <vector>
#include <string>
#include <array>

#include "dummy_helper.cuh"

#include <mpi.h>

class Lab08
{
private:
    enum class BoundaryLayerTag
    {
        LEFT = 1, RIGHT, FRONT, BACK, DOWN, UP
    };

    static const int SEND_ANY_TAG = 0;

public:
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

private:
    CudaTimer timer;
    
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
    int comm_size;

    long long block_x, block_y, block_z;
    
    bool sends_first;
    // work data
    std::vector<double> block_h;

    CudaMemory<double> block_d, prev_block_d;

    std::vector<double> left_h ,
                        right_h,
                        front_h,
                        back_h ,
                        down_h ,
                        up_h   ;

    CudaMemory<double> left_d ,
                       right_d,
                       front_d,
                       back_d ,
                       down_d ,
                       up_d   ;

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
        std::vector<double> &left_h,
        std::vector<double> &right_h,
        std::vector<double> &front_h,
        std::vector<double> &back_h,
        std::vector<double> &down_h,
        std::vector<double> &up_h,
        std::vector<MPI_Request> &receive_requests
    );

    void receive_boundary_layer(
        std::vector<double> &v, 
        int source_rank, 
        BoundaryLayerTag tag,
        std::vector<MPI_Request> &receive_requests
    );

    void send_boundary_layer(
        std::vector<double> &v_h,
        CudaMemory<double> &v_d,
        int destination_rank, 
        BoundaryLayerTag tag,
        std::vector<MPI_Request> &send_requests
    );

    void copy_boundary_layers_to_device(
        std::vector<double> &left_h,
        std::vector<double> &right_h,
        std::vector<double> &front_h,
        std::vector<double> &back_h,
        std::vector<double> &down_h,
        std::vector<double> &up_h
    );

    void set_device();

public:
    Lab08(int argc, char **argv);

    void solve();
    void write_answer();
    static void finalize();
    static void init(int argc, char **argv);
};
