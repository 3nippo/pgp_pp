#include "lab09.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <iterator>

#include "MPI_dummy_helper.hpp"

#include <mpi.h>

#include <omp.h>

Lab09::Lab09(int argc, char **argv)
{
    int initialized;

    checkMPIErrors(MPI_Initialized(
        &initialized
    ));

    if (!initialized)
        checkMPIErrors(MPI_Init(&argc, &argv));   

    checkMPIErrors(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    if (rank == 0)
        rank_0_init();
    else
        rank_non_0_init();

    block_z = rank / process_grid_shape[0] / process_grid_shape[1];
    block_y = rank % (process_grid_shape[0] * process_grid_shape[1]) / process_grid_shape[1];
    block_x = rank % (process_grid_shape[0] * process_grid_shape[1]) % process_grid_shape[1];
    
    sends_first = (block_x + block_y + block_z) % 2;

    left.resize( block_x == 0                         ? 0 : block_shape[1] * block_shape[2]);
    right.resize(block_x == process_grid_shape[0] - 1 ? 0 : block_shape[1] * block_shape[2]);
    front.resize(block_y == 0                         ? 0 : block_shape[0] * block_shape[2]);
    back.resize( block_y == process_grid_shape[1] - 1 ? 0 : block_shape[0] * block_shape[2]);
    down.resize( block_z == 0                         ? 0 : block_shape[0] * block_shape[1]);
    up.resize(   block_z == process_grid_shape[2] - 1 ? 0 : block_shape[0] * block_shape[1]);
}

void Lab09::init(int argc, char **argv)
{
    int initialized;

    checkMPIErrors(MPI_Initialized(
        &initialized
    ));

    if (!initialized)
    {
        checkMPIErrors(MPI_Init(&argc, &argv));
    }    
}

void Lab09::finalize()
{
    int finalized;

    checkMPIErrors(MPI_Finalized(
        &finalized
    ));

    if (!finalized)
    {
        checkMPIErrors(MPI_Barrier(MPI_COMM_WORLD));
        checkMPIErrors(MPI_Finalize());
    }    
}

#define locate(i, j, k) block[(i) + (j) * block_shape[0] + (k) * block_shape[0] * block_shape[1]]
#define locate_p(v, i, j, k) v[(i) + (j) * block_shape[0] + (k) * block_shape[0] * block_shape[1]]

void Lab09::rank_0_init()
{
    // input
    
    read_in_container(process_grid_shape);

    read_in_container(block_shape);

    std::cin >> output_name;

    std::cin >> eps;

    read_in_container(l);

    std::cin >> boundaries.down
             >> boundaries.up
             >> boundaries.left
             >> boundaries.right
             >> boundaries.front
             >> boundaries.back;
    
    std::cin >> u_0;

    // input done
    
    // send input data to other ranks
    
    int n_ranks = process_grid_shape[0] 
                  * process_grid_shape[1]
                  * process_grid_shape[2];

    for (int rank = 1; rank < n_ranks; ++rank)
    {
        checkMPIErrors(MPI_Send(
            process_grid_shape.data(), 
            process_grid_shape.size(), 
            MPI_LONG_LONG, 
            rank,
            SEND_ANY_TAG, 
            MPI_COMM_WORLD
        ));

        checkMPIErrors(MPI_Send(
            block_shape.data(), 
            block_shape.size(), 
            MPI_LONG_LONG, 
            rank,
            SEND_ANY_TAG, 
            MPI_COMM_WORLD
        ));

        checkMPIErrors(MPI_Send(
            output_name.data(), 
            output_name.size(), 
            MPI_CHAR, 
            rank,
            SEND_ANY_TAG, 
            MPI_COMM_WORLD
        ));

        checkMPIErrors(MPI_Send(
            &eps, 
            1, 
            MPI_DOUBLE, 
            rank,
            SEND_ANY_TAG, 
            MPI_COMM_WORLD
        ));

        checkMPIErrors(MPI_Send(
            l.data(), 
            l.size(), 
            MPI_DOUBLE, 
            rank,
            SEND_ANY_TAG, 
            MPI_COMM_WORLD
        ));

        checkMPIErrors(MPI_Send(
            reinterpret_cast<double*>(&boundaries), 
            sizeof(boundaries) / sizeof(double), 
            MPI_DOUBLE, 
            rank,
            SEND_ANY_TAG, 
            MPI_COMM_WORLD
        ));

        checkMPIErrors(MPI_Send(
            &u_0, 
            1, 
            MPI_DOUBLE, 
            rank,
            SEND_ANY_TAG, 
            MPI_COMM_WORLD
        ));
    }
}

void Lab09::rank_non_0_init() 
{
    int root_rank = 0;

    checkMPIErrors(MPI_Recv(
        process_grid_shape.data(), 
        process_grid_shape.size(), 
        MPI_LONG_LONG, 
        root_rank,
        MPI_ANY_TAG, 
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    ));

    checkMPIErrors(MPI_Recv(
        block_shape.data(), 
        block_shape.size(), 
        MPI_LONG_LONG, 
        root_rank,
        MPI_ANY_TAG, 
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    ));
    
    MPI_Status status;
    checkMPIErrors(MPI_Probe(
        root_rank, 
        MPI_ANY_TAG, 
        MPI_COMM_WORLD, 
        &status
    ));

    int output_name_count;
    checkMPIErrors(MPI_Get_count(
        &status, 
        MPI_CHAR, 
        &output_name_count
    ));

    std::vector<char> output_name_buffer(output_name_count);

    checkMPIErrors(MPI_Recv(
        output_name_buffer.data(), 
        output_name_buffer.size(), 
        MPI_CHAR, 
        root_rank,
        MPI_ANY_TAG, 
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    ));

    std::copy(
        output_name_buffer.begin(),
        output_name_buffer.end(),
        std::back_inserter(output_name)
    );

    checkMPIErrors(MPI_Recv(
        &eps, 
        1, 
        MPI_DOUBLE, 
        root_rank,
        MPI_ANY_TAG, 
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    ));

    checkMPIErrors(MPI_Recv(
        l.data(), 
        l.size(), 
        MPI_DOUBLE, 
        root_rank,
        MPI_ANY_TAG, 
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    ));

    checkMPIErrors(MPI_Recv(
        reinterpret_cast<double*>(&boundaries), 
        sizeof(boundaries) / sizeof(double), 
        MPI_DOUBLE, 
        root_rank,
        MPI_ANY_TAG, 
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    ));

    checkMPIErrors(MPI_Recv(
        &u_0, 
        1, 
        MPI_DOUBLE, 
        root_rank,
        MPI_ANY_TAG, 
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    ));
}

int Lab09::block_position_to_rank(
    long long block_x, 
    long long block_y, 
    long long block_z
)
{
    if (block_x < 0 || block_y < 0 || block_z < 0)
    {
        return -1;
    }

    if (
        block_x >= process_grid_shape[0] 
        || block_y >= process_grid_shape[1] 
        || block_z >= process_grid_shape[2]
       )
    {
        return -1;
    }


    return block_x
           + block_y * process_grid_shape[0]
           + block_z * process_grid_shape[0] * process_grid_shape[1];
}

void Lab09::send_boundary_layer(
    std::vector<double> &v, 
    int destination_rank, 
    Lab09::BoundaryLayerTag tag,
    std::vector<MPI_Request> &send_requests
)
{
    if (destination_rank > -1)
    {
        MPI_Request request;

        checkMPIErrors(MPI_Isend(
            v.data(), 
            v.size(), 
            MPI_DOUBLE, 
            destination_rank,
            static_cast<int>(tag), 
            MPI_COMM_WORLD,
            &request
        ));

        send_requests.push_back(request);
    }
}

void Lab09::send_boundary_layers(std::vector<MPI_Request> &send_requests)
{
#define fill_boundary_layer(v, outer, inner, loc) \
{ \
    if (!v.empty()) \
        for (long long i = 0; i < block_shape[outer]; ++i) \
            for (long long j = 0; j < block_shape[inner]; ++j) \
                v[i * block_shape[inner] + j] = loc; \
}
    
    fill_boundary_layer(left , 2, 1, locate(0, j, i));
    
    fill_boundary_layer(right, 2, 1, locate(block_shape[0] - 1, j, i));
    
    fill_boundary_layer(front, 2, 0, locate(j, 0, i));

    fill_boundary_layer(back , 2, 0, locate(j, block_shape[1] - 1, i));
    
    fill_boundary_layer(down , 1, 0, locate(j, i, 0));

    fill_boundary_layer(up   , 1, 0, locate(j, i, block_shape[2] - 1));

#undef fill_boundary_layer
    
    send_boundary_layer(
        left, 
        block_position_to_rank(block_x - 1, block_y, block_z),
        BoundaryLayerTag::RIGHT,
        send_requests
    );

    send_boundary_layer(
        right, 
        block_position_to_rank(block_x + 1, block_y, block_z),
        BoundaryLayerTag::LEFT,
        send_requests
    );

    send_boundary_layer(
        front, 
        block_position_to_rank(block_x, block_y - 1, block_z),
        BoundaryLayerTag::BACK,
        send_requests
    );

    send_boundary_layer(
        back, 
        block_position_to_rank(block_x, block_y + 1, block_z),
        BoundaryLayerTag::FRONT,
        send_requests
    );

    send_boundary_layer(
        down, 
        block_position_to_rank(block_x, block_y, block_z - 1),
        BoundaryLayerTag::UP,
        send_requests
    );

    send_boundary_layer(
        up, 
        block_position_to_rank(block_x, block_y, block_z + 1),
        BoundaryLayerTag::DOWN,
        send_requests
    );
}

void Lab09::receive_boundary_layer(
    std::vector<double> &v, 
    int source_rank, 
    BoundaryLayerTag tag,
    std::vector<MPI_Request> &receive_requests
)
{
    if (source_rank > -1)
    {
        MPI_Request request;

        checkMPIErrors(MPI_Irecv(
            v.data(), 
            v.size(), 
            MPI_DOUBLE, 
            source_rank,
            static_cast<int>(tag), 
            MPI_COMM_WORLD,
            &request
        ));

        receive_requests.push_back(request);
    }
}

void Lab09::receive_boundary_layers(
    std::vector<double> &left,
    std::vector<double> &right,
    std::vector<double> &front,
    std::vector<double> &back,
    std::vector<double> &down,
    std::vector<double> &up,
    std::vector<MPI_Request> &receive_requests
)
{
    receive_boundary_layer(
        left,
        block_position_to_rank(block_x - 1, block_y, block_z),
        BoundaryLayerTag::LEFT,
        receive_requests
    );

    receive_boundary_layer(
        right,
        block_position_to_rank(block_x + 1, block_y, block_z),
        BoundaryLayerTag::RIGHT,
        receive_requests
    );

    receive_boundary_layer(
        front,
        block_position_to_rank(block_x, block_y - 1, block_z),
        BoundaryLayerTag::FRONT,
        receive_requests
    );

    receive_boundary_layer(
        back,
        block_position_to_rank(block_x, block_y + 1, block_z),
        BoundaryLayerTag::BACK,
        receive_requests
    );

    receive_boundary_layer(
        down,
        block_position_to_rank(block_x, block_y, block_z - 1),
        BoundaryLayerTag::DOWN,
        receive_requests
    );

    receive_boundary_layer(
        up,
        block_position_to_rank(block_x, block_y, block_z + 1),
        BoundaryLayerTag::UP,
        receive_requests
    );
}

void Lab09::solve() 
{
    block = std::vector<double>(block_shape[0] * block_shape[1] * block_shape[2], u_0);

    std::vector<double> prev_block = block;

    std::vector<double> left (block_x == 0                         ? 0 : block_shape[1] * block_shape[2]),
                        right(block_x == process_grid_shape[0] - 1 ? 0 : block_shape[1] * block_shape[2]),
                        front(block_y == 0                         ? 0 : block_shape[0] * block_shape[2]),
                        back (block_y == process_grid_shape[1] - 1 ? 0 : block_shape[0] * block_shape[2]),
                        down (block_z == 0                         ? 0 : block_shape[0] * block_shape[1]),
                        up   (block_z == process_grid_shape[2] - 1 ? 0 : block_shape[0] * block_shape[1]);

    double n_x = block_shape[0] * process_grid_shape[0],
           n_y = block_shape[1] * process_grid_shape[1],
           n_z = block_shape[2] * process_grid_shape[2];

    double h_x_pow_minus_2 = n_x * n_x / l[0] / l[0],
           h_y_pow_minus_2 = n_y * n_y / l[1] / l[1],
           h_z_pow_minus_2 = n_z * n_z / l[2] / l[2],
           denominator = 2 * (h_x_pow_minus_2 + h_y_pow_minus_2 + h_z_pow_minus_2);
                                       
    std::vector<MPI_Request> send_requests,
                             receive_requests;

    while (true)
    {
        if (sends_first)
        {
            send_boundary_layers(send_requests);

            checkMPIErrors(MPI_Waitall(
                send_requests.size(),
                send_requests.data(),
                MPI_STATUSES_IGNORE
            ));

            send_requests.clear();
            
            receive_boundary_layers(
                left,
                right,
                front,
                back,
                down,
                up,
                receive_requests
            );

            checkMPIErrors(MPI_Waitall(
                receive_requests.size(),
                receive_requests.data(),
                MPI_STATUSES_IGNORE
            ));

            receive_requests.clear();
        }
        else
        {
            receive_boundary_layers(
                left,
                right,
                front,
                back,
                down,
                up,
                receive_requests
            );

            checkMPIErrors(MPI_Waitall(
                receive_requests.size(),
                receive_requests.data(),
                MPI_STATUSES_IGNORE
            ));

            receive_requests.clear();

            send_boundary_layers(send_requests);

            checkMPIErrors(MPI_Waitall(
                send_requests.size(),
                send_requests.data(),
                MPI_STATUSES_IGNORE
            ));

            send_requests.clear();
        }

        double max_abs_difference = 0;
        
        #pragma omp parallel for \
            schedule(static) \
            collapse(3) \
            reduction(max : max_abs_difference)
        for (long long i = 0; i < block_shape[0]; ++i)
            for (long long j = 0; j < block_shape[1]; ++j)
                for (long long k = 0; k < block_shape[2]; ++k)
                {
                    double u_left  = i == 0                  ? (left.empty()  ? boundaries.left  : left[ j + block_shape[1] * k]) : locate_p(prev_block, i - 1, j, k),
                           u_right = i == block_shape[0] - 1 ? (right.empty() ? boundaries.right : right[j + block_shape[1] * k]) : locate_p(prev_block, i + 1, j, k),
                           u_front = j == 0                  ? (front.empty() ? boundaries.front : front[i + block_shape[0] * k]) : locate_p(prev_block, i, j - 1, k),
                           u_back  = j == block_shape[1] - 1 ? (back.empty()  ? boundaries.back  : back[ i + block_shape[0] * k]) : locate_p(prev_block, i, j + 1, k),
                           u_down  = k == 0                  ? (down.empty()  ? boundaries.down  : down[ i + block_shape[0] * j]) : locate_p(prev_block, i, j, k - 1),
                           u_up    = k == block_shape[2] - 1 ? (up.empty()    ? boundaries.up    : up[   i + block_shape[0] * j]) : locate_p(prev_block, i, j, k + 1);
                    
                    locate(i, j, k) =  (u_left  + u_right) * h_x_pow_minus_2;

                    locate(i, j, k) += (u_front + u_back ) * h_y_pow_minus_2;

                    locate(i, j, k) += (u_down  + u_up   ) * h_z_pow_minus_2;

                    locate(i, j, k) /= denominator;

                    max_abs_difference = std::max(std::abs(locate(i, j, k) - locate_p(prev_block, i, j, k)), max_abs_difference);
                }   
        
        double total_max_abs_difference;

        checkMPIErrors(MPI_Allreduce(
            &max_abs_difference,
            &total_max_abs_difference,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            MPI_COMM_WORLD
        ));

        if (total_max_abs_difference < eps)
            break;

        prev_block = block; 
    }
}

void Lab09::write_answer()
{
    std::ofstream output;

    if (rank == 0)
    {
        output.open(output_name, std::ofstream::trunc);

        output << std::scientific
               << std::setprecision(6);
    }

    std::vector<double> buffer(block_shape[0]);
    
    for (long long k = 0; k < process_grid_shape[2]; ++k)
        for (long long height = 0; height < block_shape[2]; ++height)
            for (long long j = 0; j < process_grid_shape[1]; ++j)
                for (long long row = 0; row < block_shape[1]; ++row)        
                    for (long long i = 0; i < process_grid_shape[0]; ++i)
                    {
                        int sender_rank = block_position_to_rank(i, j, k);

                        if (sender_rank == 0 && rank == 0)
                        {
                            for (long long column = 0; column < block_shape[0] - 1; ++column)
                                output << locate(column, row, height) << " ";

                            output << locate(block_shape[0] - 1, row, height);
                        }
                        else
                        {
                            if (rank == 0)
                            {
                                checkMPIErrors(MPI_Recv(
                                    buffer.data(), 
                                    block_shape[0], 
                                    MPI_DOUBLE, 
                                    sender_rank,
                                    MPI_ANY_TAG, 
                                    MPI_COMM_WORLD,
                                    MPI_STATUS_IGNORE
                                ));

                                for (long long column = 0; column < block_shape[0] - 1; ++column)
                                    output << buffer[column] << " ";

                                output << buffer[block_shape[0] - 1];
                            }
                            else if (rank == sender_rank)
                            {
                                checkMPIErrors(MPI_Send(
                                    &locate(0, row, height), 
                                    block_shape[0], 
                                    MPI_DOUBLE,
                                    0,
                                    SEND_ANY_TAG, 
                                    MPI_COMM_WORLD
                                ));
                            }
                        }
                        
                        if (rank == 0)
                        {
                            if (i == process_grid_shape[0] - 1)
                            {
                                output << "\n";

                                if (row == block_shape[1] - 1 && j == process_grid_shape[1] - 1
                                    && (height != block_shape[2] - 1 || k != process_grid_shape[2] - 1)
                                )
                                    output << "\n";
                            }
                            else
                                output << " ";
                        }
                    }
}
