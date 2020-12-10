#include "lab08.cuh"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <iterator>

#include <cstdio>

#include "MPI_dummy_helper.hpp"
#include "dummy_helper.cuh"

#include <mpi.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define GRID_SIZE 1
#define BLOCK_SIZE 4

#define GRID_SIZE_dim3 dim3(GRID_SIZE, GRID_SIZE, GRID_SIZE)
#define BLOCK_SIZE_dim3 dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)

#define locate(i, j, k) block_h[(i) + (j) * block_shape[0] + (k) * block_shape[0] * block_shape[1]]
#define locate_p(v, i, j, k) v[(i) + (j) * block_shape[0] + (k) * block_shape[0] * block_shape[1]]

__global__
void init_array(
    double *v,
    long long count,
    double init_value
)
{
    const long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    const long long idy = blockDim.y * blockIdx.y + threadIdx.y;
    const long long idz = blockDim.z * blockIdx.z + threadIdx.z;

    const long long id = idx + idy * blockDim.x * gridDim.x + idz * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    const long long offset = gridDim.x * blockDim.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z;

    for (long long i = id; i < count; i += offset)
        v[i] = init_value;
}

void Lab08::set_device()
{
    int device_count;

    checkCudaErrors(cudaGetDeviceCount(&device_count));

    checkCudaErrors(cudaSetDevice(rank % device_count));
}

Lab08::Lab08(int argc, char **argv)
{
    init(argc, argv);

    checkMPIErrors(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    set_device();
    
    // read input data
    if (rank == 0)
        rank_0_init();
    else
        rank_non_0_init();

    block_z = rank / process_grid_shape[0] / process_grid_shape[1];
    block_y = rank % (process_grid_shape[0] * process_grid_shape[1]) / process_grid_shape[0];
    block_x = rank % (process_grid_shape[0] * process_grid_shape[1]) % process_grid_shape[0];
    
    sends_first = (block_x + block_y + block_z) % 2;

    CudaKernelChecker kernel_checker;

    auto block_init_routine =
        [&kernel_checker, this](CudaMemory<double> &v, const char* kernel_name)
        {
            v.alloc(block_shape[0] * block_shape[1] * block_shape[2]);

            init_array<<<GRID_SIZE_dim3, BLOCK_SIZE_dim3>>>(
                v.get(),
                block_shape[0] * block_shape[1] * block_shape[2],
                u_0
            );

            kernel_checker.check(kernel_name);
        };
    
    block_init_routine(block_d, "init block_d");
    block_init_routine(prev_block_d, "init prev_block_d");

    auto boundary_layer_init_routine = 
        [this](
            std::vector<double> &v_h,
            CudaMemory<double> &v_d,
            const bool layer_not_needed, 
            const long long count
        )
        {
            v_h.resize( layer_not_needed ? 0 : count);
            v_d.alloc(  layer_not_needed ? 0 : count);
        };

    boundary_layer_init_routine(
        left_h,
        left_d,
        block_x == 0,
        block_shape[1] * block_shape[2]
    );

    boundary_layer_init_routine(
        right_h,
        right_d,
        block_x == process_grid_shape[0] - 1,
        block_shape[1] * block_shape[2]
    );
    
    boundary_layer_init_routine(
        front_h,
        front_d,
        block_y == 0,
        block_shape[0] * block_shape[2]
    );

    boundary_layer_init_routine(
        back_h,
        back_d,
        block_y == process_grid_shape[1] - 1,
        block_shape[0] * block_shape[2]
    );
    
    boundary_layer_init_routine(
        down_h,
        down_d,
        block_z == 0,
        block_shape[0] * block_shape[1]
    );

    boundary_layer_init_routine(
        up_h,
        up_d,
        block_z == process_grid_shape[2] - 1,
        block_shape[0] * block_shape[1]
    );

    timer.start();
}

void Lab08::init(int argc, char **argv)
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

void Lab08::finalize()
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

void Lab08::rank_0_init()
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

void Lab08::rank_non_0_init() 
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

int Lab08::block_position_to_rank(
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

void Lab08::send_boundary_layer(
    std::vector<double> &v_h,
    CudaMemory<double> &v_d,
    int destination_rank, 
    Lab08::BoundaryLayerTag tag,
    std::vector<MPI_Request> &send_requests
)
{
    if (destination_rank > -1)
    {
        v_d.memcpy(
            v_h.data(),
            cudaMemcpyDeviceToHost
        );

        MPI_Request request;

        checkMPIErrors(MPI_Isend(
            v_h.data(), 
            v_h.size(), 
            MPI_DOUBLE, 
            destination_rank,
            static_cast<int>(tag), 
            MPI_COMM_WORLD,
            &request
        ));

        send_requests.push_back(request);
    }
}

void Lab08::send_boundary_layers(std::vector<MPI_Request> &send_requests)
{
    send_boundary_layer(
        left_h, 
        left_d,
        block_position_to_rank(block_x - 1, block_y, block_z),
        BoundaryLayerTag::RIGHT,
        send_requests
    );

    send_boundary_layer(
        right_h,
        right_d,
        block_position_to_rank(block_x + 1, block_y, block_z),
        BoundaryLayerTag::LEFT,
        send_requests
    );

    send_boundary_layer(
        front_h, 
        front_d, 
        block_position_to_rank(block_x, block_y - 1, block_z),
        BoundaryLayerTag::BACK,
        send_requests
    );

    send_boundary_layer(
        back_h, 
        back_d, 
        block_position_to_rank(block_x, block_y + 1, block_z),
        BoundaryLayerTag::FRONT,
        send_requests
    );

    send_boundary_layer(
        down_h, 
        down_d, 
        block_position_to_rank(block_x, block_y, block_z - 1),
        BoundaryLayerTag::UP,
        send_requests
    );

    send_boundary_layer(
        up_h, 
        up_d, 
        block_position_to_rank(block_x, block_y, block_z + 1),
        BoundaryLayerTag::DOWN,
        send_requests
    );
}

void Lab08::receive_boundary_layer(
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

void Lab08::receive_boundary_layers(
    std::vector<double> &left_h,
    std::vector<double> &right_h,
    std::vector<double> &front_h,
    std::vector<double> &back_h,
    std::vector<double> &down_h,
    std::vector<double> &up_h,
    std::vector<MPI_Request> &receive_requests
)
{
    receive_boundary_layer(
        left_h,
        block_position_to_rank(block_x - 1, block_y, block_z),
        BoundaryLayerTag::LEFT,
        receive_requests
    );

    receive_boundary_layer(
        right_h,
        block_position_to_rank(block_x + 1, block_y, block_z),
        BoundaryLayerTag::RIGHT,
        receive_requests
    );

    receive_boundary_layer(
        front_h,
        block_position_to_rank(block_x, block_y - 1, block_z),
        BoundaryLayerTag::FRONT,
        receive_requests
    );

    receive_boundary_layer(
        back_h,
        block_position_to_rank(block_x, block_y + 1, block_z),
        BoundaryLayerTag::BACK,
        receive_requests
    );

    receive_boundary_layer(
        down_h,
        block_position_to_rank(block_x, block_y, block_z - 1),
        BoundaryLayerTag::DOWN,
        receive_requests
    );

    receive_boundary_layer(
        up_h,
        block_position_to_rank(block_x, block_y, block_z + 1),
        BoundaryLayerTag::UP,
        receive_requests
    );
}

__device__ long long block_shape[3];

__global__
void copy_block_to_prev_block(
    double *block,
    double *prev_block
)
{
    const long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    const long long idy = blockDim.y * blockIdx.y + threadIdx.y;
    const long long idz = blockDim.z * blockIdx.z + threadIdx.z;
    
    const long long offset_x = blockDim.x * gridDim.x;
    const long long offset_y = blockDim.y * gridDim.y;
    const long long offset_z = blockDim.z * gridDim.z;
    
    for (long long k = idz; k < block_shape[2]; k += offset_z)
        for (long long j = idy; j < block_shape[1]; j += offset_y)
            for (long long i = idx; i < block_shape[0]; i += offset_x)
                locate_p(prev_block, i, j, k) = locate_p(block, i, j, k);
}

__global__
void prev_block_to_abs_difference_array(
    double *block,
    double *prev_block
)
{
    const long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    const long long idy = blockDim.y * blockIdx.y + threadIdx.y;
    const long long idz = blockDim.z * blockIdx.z + threadIdx.z;
    
    const long long offset_x = blockDim.x * gridDim.x;
    const long long offset_y = blockDim.y * gridDim.y;
    const long long offset_z = blockDim.z * gridDim.z;
    
    for (long long k = idz; k < block_shape[2]; k += offset_z)
        for (long long j = idy; j < block_shape[1]; j += offset_y)
            for (long long i = idx; i < block_shape[0]; i += offset_x)
                locate_p(prev_block, i, j, k) = fabs(locate_p(block, i, j, k) - locate_p(prev_block, i, j, k));
}

__global__
void block_iter_process(
    double *block,
    double *prev_block,
    double *left,
    double *right,
    double *front,
    double *back,
    double *down,
    double *up,
    Lab08::Boundaries boundaries,
    double h_x_pow_minus_2,
    double h_y_pow_minus_2,
    double h_z_pow_minus_2,
    double denominator
)
{
    const long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    const long long idy = blockDim.y * blockIdx.y + threadIdx.y;
    const long long idz = blockDim.z * blockIdx.z + threadIdx.z;
    
    const long long offset_x = blockDim.x * gridDim.x;
    const long long offset_y = blockDim.y * gridDim.y;
    const long long offset_z = blockDim.z * gridDim.z;
    
    for (long long k = idz; k < block_shape[2]; k += offset_z)
        for (long long j = idy; j < block_shape[1]; j += offset_y)
            for (long long i = idx; i < block_shape[0]; i += offset_x)
            {
                double u_left  = i == 0                  ? (left  == nullptr ? boundaries.left  : left[ j + block_shape[1] * k]) : locate_p(prev_block, i - 1, j, k),
                       u_right = i == block_shape[0] - 1 ? (right == nullptr ? boundaries.right : right[j + block_shape[1] * k]) : locate_p(prev_block, i + 1, j, k),
                       u_front = j == 0                  ? (front == nullptr ? boundaries.front : front[i + block_shape[0] * k]) : locate_p(prev_block, i, j - 1, k),
                       u_back  = j == block_shape[1] - 1 ? (back  == nullptr ? boundaries.back  : back[ i + block_shape[0] * k]) : locate_p(prev_block, i, j + 1, k),
                       u_down  = k == 0                  ? (down  == nullptr ? boundaries.down  : down[ i + block_shape[0] * j]) : locate_p(prev_block, i, j, k - 1),
                       u_up    = k == block_shape[2] - 1 ? (up    == nullptr ? boundaries.up    : up[   i + block_shape[0] * j]) : locate_p(prev_block, i, j, k + 1);
                
                locate_p(block, i, j, k) =  (u_left  + u_right) * h_x_pow_minus_2;

                locate_p(block, i, j, k) += (u_front + u_back ) * h_y_pow_minus_2;

                locate_p(block, i, j, k) += (u_down  + u_up   ) * h_z_pow_minus_2;

                locate_p(block, i, j, k) /= denominator;
            }
}

__global__
void init_boundary_layers(
    double *block,
    double *left,
    double *right,
    double *front,
    double *back,
    double *down,
    double *up
)
{
    const long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    const long long idy = blockDim.y * blockIdx.y + threadIdx.y;
    const long long idz = blockDim.z * blockIdx.z + threadIdx.z;
    
    const long long offset_x = blockDim.x * gridDim.x;
    const long long offset_y = blockDim.y * gridDim.y;
    const long long offset_z = blockDim.z * gridDim.z;

#define fill_boundary_layer(v, outer, inner, outer_start, inner_start, outer_offset, inner_offset, loc) \
{ \
    if (v) \
    { \
        for (long long i = outer_start; i < block_shape[outer]; i += outer_offset) \
            for (long long j = inner_start; j < block_shape[inner]; j += inner_offset) \
                v[i * block_shape[inner] + j] = loc; \
    } \
} \

    if (idx == 0)
        fill_boundary_layer(left , 2, 1, idz, idy, offset_z, offset_y, locate_p(block, 0, j, i));
    
    if (idx == 1)
        fill_boundary_layer(right, 2, 1, idz, idy, offset_z, offset_y, locate_p(block, block_shape[0] - 1, j, i));
    
    if (idy == 0)
        fill_boundary_layer(front, 2, 0, idz, idx, offset_z, offset_x, locate_p(block, j, 0, i));

    if (idy == 1)
        fill_boundary_layer(back , 2, 0, idz, idx, offset_z, offset_x, locate_p(block, j, block_shape[1] - 1, i));
    
    if (idz == 0)
        fill_boundary_layer(down , 1, 0, idy, idx, offset_y, offset_x, locate_p(block, j, i, 0));

    if (idz == 1)
        fill_boundary_layer(up   , 1, 0, idy, idx, offset_y, offset_x, locate_p(block, j, i, block_shape[2] - 1));

#undef fill_boundary_layer
}

void Lab08::copy_boundary_layers_to_device(
    std::vector<double> &left_h,
    std::vector<double> &right_h,
    std::vector<double> &front_h,
    std::vector<double> &back_h,
    std::vector<double> &down_h,
    std::vector<double> &up_h
)
{
    auto copy_boundary_layer_to_device = 
        [](
            std::vector<double> &v_h,
            CudaMemory<double> &v_d
        )
        {
            v_d.memcpy(v_h.data(), cudaMemcpyHostToDevice);
        };

    copy_boundary_layer_to_device(left_h, left_d);
    copy_boundary_layer_to_device(right_h, right_d);
    copy_boundary_layer_to_device(front_h, front_d);
    copy_boundary_layer_to_device(back_h, back_d);
    copy_boundary_layer_to_device(down_h, down_d);
    copy_boundary_layer_to_device(up_h, up_d);
}

void Lab08::solve() 
{
    std::vector<double> left_h (block_x == 0                         ? 0 : block_shape[1] * block_shape[2]),
                        right_h(block_x == process_grid_shape[0] - 1 ? 0 : block_shape[1] * block_shape[2]),
                        front_h(block_y == 0                         ? 0 : block_shape[0] * block_shape[2]),
                        back_h (block_y == process_grid_shape[1] - 1 ? 0 : block_shape[0] * block_shape[2]),
                        down_h (block_z == 0                         ? 0 : block_shape[0] * block_shape[1]),
                        up_h   (block_z == process_grid_shape[2] - 1 ? 0 : block_shape[0] * block_shape[1]);

    double n_x = block_shape[0] * process_grid_shape[0],
           n_y = block_shape[1] * process_grid_shape[1],
           n_z = block_shape[2] * process_grid_shape[2];

    double h_x_pow_minus_2 = n_x * n_x / l[0] / l[0],
           h_y_pow_minus_2 = n_y * n_y / l[1] / l[1],
           h_z_pow_minus_2 = n_z * n_z / l[2] / l[2],
           denominator = 2 * (h_x_pow_minus_2 + h_y_pow_minus_2 + h_z_pow_minus_2);
                                       
    std::vector<MPI_Request> send_requests,
                             receive_requests;
    
    checkCudaErrors(cudaMemcpyToSymbol(
        ::block_shape,
        block_shape.data(),
        block_shape.size() * sizeof(decltype(block_shape[0])),
        0,
        cudaMemcpyHostToDevice
    ));

    thrust::device_ptr<double> abs_difference_array = thrust::device_pointer_cast(prev_block_d.get());

    while (true)
    {
        CudaKernelChecker checker;
        
        init_boundary_layers<<<GRID_SIZE_dim3, BLOCK_SIZE_dim3>>>(
            block_d.get(),
            left_d.get(),
            right_d.get(),
            front_d.get(),
            back_d.get(),
            down_d.get(),
            up_d.get()   
        );

        checker.check("init_boundary_layers");
        
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
                left_h,
                right_h,
                front_h,
                back_h,
                down_h,
                up_h,
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
                left_h,
                right_h,
                front_h,
                back_h,
                down_h,
                up_h,
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

        copy_boundary_layers_to_device(
            left_h,
            right_h,
            front_h,
            back_h,
            down_h,
            up_h
        );
        
        copy_block_to_prev_block<<<GRID_SIZE_dim3, BLOCK_SIZE_dim3>>>(
            block_d.get(),
            prev_block_d.get()
        );

        checker.check("copy_block_to_prev_block");

        block_iter_process<<<GRID_SIZE_dim3, BLOCK_SIZE_dim3>>>(
            block_d.get(),
            prev_block_d.get(),
            left_d.get(),
            right_d.get(),
            front_d.get(),
            back_d.get(),
            down_d.get(),
            up_d.get(),
            boundaries,
            h_x_pow_minus_2,
            h_y_pow_minus_2,
            h_z_pow_minus_2,
            denominator
        );
        
        checker.check("iter process kernel");

        prev_block_to_abs_difference_array<<<GRID_SIZE_dim3, BLOCK_SIZE_dim3>>>(
            block_d.get(),
            prev_block_d.get()
        );

        checker.check("prev_block_to_abs_difference_array");
        
        double max_abs_difference = *thrust::max_element(
            abs_difference_array,
            abs_difference_array + prev_block_d.count
        );

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
    }
}

void Lab08::write_answer()
{
    timer.stop();

    if (rank == 0)
        timer.print_time();

    MPI_File file;
    
    int delete_error = MPI_File_delete(output_name.c_str(), MPI_INFO_NULL);
    
    if (delete_error != 0 && delete_error != MPI_ERR_NO_SUCH_FILE)
        checkMPIErrors(delete_error);
	
    checkMPIErrors(MPI_File_open(
        MPI_COMM_WORLD, 
        output_name.c_str(),
        MPI_MODE_CREATE | MPI_MODE_WRONLY, 
        MPI_INFO_NULL, 
        &file
    ));
    
    // create type

    MPI_Datatype Number;
    const int number_chars_count = 16; // 0 . 000000 e+000 ' '
    
    checkMPIErrors(MPI_Type_contiguous(
        number_chars_count,
        MPI_CHAR,
        &Number
    ));

    MPI_Datatype BlockRow;

    checkMPIErrors(MPI_Type_contiguous(
        block_shape[0],
        Number,
        &BlockRow
    ));
    
    MPI_Datatype BlockPlane;
    
    std::vector<int> BlockPlane_blocklengths;
    std::vector<int> BlockPlane_displacements;

    for (size_t i = 0; i < block_shape[1]; ++i)
    {
        BlockPlane_blocklengths.push_back(1);
        BlockPlane_displacements.push_back(i * process_grid_shape[0]);
    }

    checkMPIErrors(MPI_Type_create_hvector(block_shape[1], 1, process_grid_shape[0] * block_shape[0] * number_chars_count, BlockRow, &BlockPlane));
    /* checkMPIErrors(MPI_Type_indexed( */
    /*     block_shape[1], */
    /*     BlockPlane_blocklengths.data(), */
    /*     BlockPlane_displacements.data(), */
    /*     BlockRow, */
    /*     &BlockPlane */
    /* )); */

    MPI_Datatype Block;
    
    std::vector<int> Block_blocklengths;
    std::vector<int> Block_displacements;
    
    for (size_t i = 0; i < block_shape[2]; ++i)
    {
        Block_blocklengths.push_back(1);
        Block_displacements.push_back(i * process_grid_shape[1]);
    }

    checkMPIErrors(MPI_Type_create_hvector(block_shape[2], 1, process_grid_shape[1] * block_shape[0] * number_chars_count * process_grid_shape[0] * block_shape[1], BlockPlane, &Block));
    /* checkMPIErrors(MPI_Type_indexed( */
    /*     block_shape[2], */
    /*     Block_blocklengths.data(), */
    /*     Block_displacements.data(), */
    /*     BlockPlane, */
    /*     &Block */
    /* )); */
    checkMPIErrors(MPI_Type_commit(&Block));
    
    // set view with created type

    MPI_Offset offset = 0;

    offset += block_shape[0] * number_chars_count * block_x;
    offset += block_shape[0] * block_shape[1] * process_grid_shape[0] * number_chars_count * block_y;
    offset += block_shape[0] * block_shape[1] * block_shape[2] * process_grid_shape[0] * process_grid_shape[1] * number_chars_count * block_z;
    
    checkMPIErrors(MPI_File_set_view(
        file,
        offset,
        MPI_CHAR,
        Block,
        "native",
        MPI_INFO_NULL
    ));

    // create buffer with data to write

    std::string buffer;
    size_t buffer_pos = 0;

    block_h.resize(block_d.count);
    block_d.memcpy(block_h.data(), cudaMemcpyDeviceToHost);
    
    /* for (size_t i = 0; i < block_h.size(); ++i) */
    /* { */
    /*     std::cout << i << std::endl; */
    /*     block_h[i] = i + block_h.size() * rank; */
    /* } */

    for (long long k = 0; k < block_shape[2]; ++k)
        for (long long j = 0; j < block_shape[1]; ++j)
            for (long long i = 0; i < block_shape[0]; ++i)
            {
                buffer.resize(buffer_pos + number_chars_count);
                
                sprintf(&buffer[buffer_pos], "%-16e", locate(i, j, k));
                
                buffer_pos += number_chars_count;
                
                if (block_x == process_grid_shape[0] - 1 && i == block_shape[0] - 1)
                {
                    buffer[buffer_pos - 1] = '\n';

                    if (j == block_shape[1] - 1 && block_y == process_grid_shape[1] - 1 && (block_z != process_grid_shape[2] - 1 || k != block_shape[2] - 1))
                        buffer[buffer_pos - 2] = '\n';
                }    
            }
    
    /* buffer[0] = 'R'; */
    /* buffer[1] = 'a'; */
    /* buffer[2] = 'n'; */
    /* buffer[3] = 'k'; */
    /* buffer[4] = ' '; */
    /* buffer[5] = '0' + rank; */

    /* if (rank == 0) { */
    /*     std::cout << buffer << std::endl; */
    /*     for (int i = 1; i < 8; i++) { */
    /*         MPI_Recv(&buffer[0], buffer.size(), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); */
    /*         std::cout << std::endl << buffer << std::endl; */
    /*     } */
    /* } else { */
    /*     MPI_Send(&buffer[0], buffer.size(), MPI_CHAR, 0, 0, MPI_COMM_WORLD); */
    /* } */
    // write data from buffer
    
    checkMPIErrors(MPI_File_write_all(
        file, 
        buffer.data(), 
        buffer.size(),
        MPI_CHAR, 
        MPI_STATUS_IGNORE
    ));

    // close file

    checkMPIErrors(MPI_File_close(&file));
}
