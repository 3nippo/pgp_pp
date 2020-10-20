#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <stdexcept>

#include "MPI_dummy_helper.hpp"

#include <mpi.h>

#define TEST_SOMETHING
#define TEST_INPUT

#ifdef TEST_SOMETHING
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#endif

namespace
{
    struct Boundaries
    {
        double up;
        double down;
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

    using BlockType = std::vector< std::vector< std::vector<double> > >;

    int SEND_ANY_TAG = 1;

    #ifdef TEST_INPUT
    std::array<long long, 3> g_process_grid_shape;
    std::array<long long, 3> g_block_shape;
    std::string g_output_name;
    double g_eps;
    std::array<double, 3> g_l;
    Boundaries g_boundaries;
    double g_u_0;

    class InputEnvironment : public ::testing::Environment 
    {
    public:
        void SetData(
            const std::array<long long, 3> &process_grid_shape,
            const std::array<long long, 3> &block_shape,
            const std::string &output_name,
            const double &eps,
            const std::array<double, 3> &l,
            const Boundaries &boundaries,
            const double &u_0
        )
        {
            g_process_grid_shape = process_grid_shape;
            g_block_shape = block_shape;
            g_output_name = output_name;
            g_eps = eps;
            g_l = l;
            g_boundaries = boundaries;
            g_u_0 = u_0;
        }

        void SetUp() override {}

        virtual ~InputEnvironment() {}

        // Override this to define how to tear down the environment.
        void TearDown() override {}
    };
    #endif
};


#ifdef TEST_INPUT
TEST (InputTest, Main)
{
    int rank;

    checkMPIErrors(MPI_Comm_rank(MPI_COMM_WORLD, &rank));   
    
    EXPECT_THAT(
        g_process_grid_shape,
        ::testing::ElementsAreArray({1ll, 1ll, 2ll})
    )
    << "Wrong process_shape_grid\n" << "Rank: " << rank;

    EXPECT_THAT(
        g_block_shape,
        ::testing::ElementsAreArray({3ll, 3ll, 3ll})
    )
    << "Wrong block_shape\n" << "Rank: " << rank;

    EXPECT_EQ(g_output_name, "mpi.out") << "Wrong output_name\n" << "Rank: " << rank;

    EXPECT_NEAR(g_eps, 1e-10, 1e-11) << "Wrong eps\n" << "Rank: " << rank;

    EXPECT_THAT(
        g_l,
        ::testing::ElementsAreArray({1.0, 1.0, 2.0})
    ) 
    << "Wrong l\n" << "Rank: " << rank;

    EXPECT_EQ(
        g_boundaries,
        Boundaries({7.0, 0.0, 5.0, 0.0, 3.0, 0.0})
    )
    << "Wrong boundaries\n" << "Rank: " << rank;

    EXPECT_EQ(g_u_0, 5.0) << "Wrong u_0\n" << "Rank: " << rank;
}
#endif


void rank_0_init(
    std::array<long long, 3> &process_grid_shape,
    std::array<long long, 3> &block_shape,
    std::string &output_name,
    double &eps,
    std::array<double, 3> &l,
    Boundaries &boundaries,
    double &u_0
)
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
    
    long long n_ranks = process_grid_shape[0] 
                      * process_grid_shape[1]
                      * process_grid_shape[2];

    for (long long rank = 1; rank < n_ranks; ++rank)
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

void rank_non_0_init(
    std::array<long long, 3> &process_grid_shape,
    std::array<long long, 3> &block_shape,
    std::string &output_name,
    double &eps,
    std::array<double, 3> &l,
    Boundaries &boundaries,
    double &u_0
) 
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

    output_name.resize(output_name_count);

    checkMPIErrors(MPI_Recv(
        &output_name[0], 
        output_name.size(), 
        MPI_CHAR, 
        root_rank,
        MPI_ANY_TAG, 
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    ));

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

void iter_process(
    std::array<long long, 3> &process_grid_shape,
    std::array<long long, 3> &block_shape,
    std::string &output_name,
    double &eps,
    std::array<double, 3> &l,
    Boundaries &boundaries,
    BlockType &block
) 
{
    BlockType prev_block = block;


}

int submain(int argc, char **argv)
{
    #ifdef TEST_SOMETHING
    ::testing::InitGoogleTest(&argc, argv);
    #endif

    checkMPIErrors(MPI_Init(&argc, &argv));
    
    int rank;

    checkMPIErrors(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    
    // resources

    std::array<long long, 3> process_grid_shape;

    std::array<long long, 3> block_shape;

    std::string output_name;

    double eps;

    std::array<double, 3> l;

    Boundaries boundaries;

    double u_0;

    // resources end

    if (rank == 0)
        rank_0_init(
            process_grid_shape,
            block_shape,
            output_name,
            eps,
            l,
            boundaries,
            u_0
        );
    else
        rank_non_0_init(
            process_grid_shape,
            block_shape,
            output_name,
            eps,
            l,
            boundaries,
            u_0
        );

    int error_code = 0;

    #ifdef TEST_INPUT
    ::testing::GTEST_FLAG(filter) = "InputTest*";
    
    ::testing::Environment* const env =
        ::testing::AddGlobalTestEnvironment(new InputEnvironment);

    dynamic_cast<InputEnvironment*>(env)->SetData(
        process_grid_shape,
        block_shape,
        output_name,
        eps,
        l,
        boundaries,
        u_0
    );

    error_code = RUN_ALL_TESTS(); 
    #endif

    std::vector< std::vector< std::vector<double> > > block(
        block_shape[0],
        std::vector< std::vector<double> >(
            block_shape[1],
            std::vector<double>(block_shape[2], u_0)
        )
    );

    iter_process(
        process_grid_shape,
        block_shape,
        output_name,
        eps,
        l,
        boundaries,
        block
    );

    checkMPIErrors(MPI_Finalize());

    return error_code;
}

int main(int argc, char **argv)
{
    try
    {
        return submain(argc, argv);
    }
    catch (const std::exception &err)
    {
        std::cerr << "ERROR:\n\n" << err.what() << std::endl;
        return EXIT_FAILURE;
    }
}
