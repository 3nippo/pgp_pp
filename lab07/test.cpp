#include <unistd.h>
#include <stdexcept>
#include <iostream>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "lab07.public.hpp"
#include "lab07cpu.public.hpp"
#include "MPI_dummy_helper.hpp"

#include <chrono>

#include <mpi.h>

int argc_;
char **argv_;

TEST (MPIInputTest, t1)
{
    Lab07 solver(argc_, argv_);
    
    EXPECT_THAT(
        solver.process_grid_shape,
        ::testing::ElementsAreArray({1, 1, 1})
    )
    << "Wrong process_shape_grid\n" << "Rank: " << solver.rank;

    EXPECT_THAT(
        solver.block_shape,
        ::testing::ElementsAreArray({3, 3, 3})
    )
    << "Wrong block_shape\n" << "Rank: " << solver.rank;

    EXPECT_EQ(solver.output_name, "mpi.out") << "Wrong output_name\n" << "Rank: " << solver.rank;

    EXPECT_NEAR(solver.eps, 1e-10, 1e-11) << "Wrong eps\n" << "Rank: " << solver.rank;

    EXPECT_THAT(
        solver.l,
        ::testing::ElementsAreArray({1.0, 1.0, 1.0})
    ) 
    << "Wrong l\n" << "Rank: " << solver.rank;

    EXPECT_EQ(
        solver.boundaries,
        Lab07::Boundaries({7.0, 7.0, 7.0, 7.0, 7.0, 7.0})
    )
    << "Wrong boundaries\n" << "Rank: " << solver.rank;

    EXPECT_EQ(solver.u_0, 0.0) << "Wrong u_0\n" << "Rank: " << solver.rank;
}

TEST (MPIInputTest, t2)
{
    Lab07 solver(argc_, argv_);
    
    EXPECT_THAT(
        solver.process_grid_shape,
        ::testing::ElementsAreArray({1, 1, 2})
    )
    << "Wrong process_shape_grid\n" << "Rank: " << solver.rank;

    EXPECT_THAT(
        solver.block_shape,
        ::testing::ElementsAreArray({3, 3, 3})
    )
    << "Wrong block_shape\n" << "Rank: " << solver.rank;

    EXPECT_EQ(solver.output_name, "mpi.out") << "Wrong output_name\n" << "Rank: " << solver.rank;

    EXPECT_NEAR(solver.eps, 1e-10, 1e-11) << "Wrong eps\n" << "Rank: " << solver.rank;

    EXPECT_THAT(
        solver.l,
        ::testing::ElementsAreArray({1.0, 1.0, 2.0})
    ) 
    << "Wrong l\n" << "Rank: " << solver.rank;

    EXPECT_EQ(
        solver.boundaries,
        Lab07::Boundaries({7.0, 0.0, 5.0, 0.0, 3.0, 0.0})
    )
    << "Wrong boundaries\n" << "Rank: " << solver.rank;

    EXPECT_EQ(solver.u_0, 5.0) << "Wrong u_0\n" << "Rank: " << solver.rank;
}

TEST (MPIOutputTest, t1)
{
    Lab07 solver(argc_, argv_);

    solver.solve();
    solver.write_answer();

    int cmp_result = system("cmp t1.out mpi.out");

    EXPECT_EQ(cmp_result, 0);
}

TEST (MPIOutputTest, t2)
{
    Lab07 solver(argc_, argv_);

    solver.solve();
    solver.write_answer();

    int cmp_result = system("cmp t2.out mpi.out");

    EXPECT_EQ(cmp_result, 0);
}

TEST (CpuInputTest, t1)
{
    Lab07cpu solver(argc_, argv_);
    
    EXPECT_THAT(
        solver.process_grid_shape,
        ::testing::ElementsAreArray({1, 1, 1})
    )
    << "Wrong process_shape_grid";

    EXPECT_THAT(
        solver.block_shape,
        ::testing::ElementsAreArray({3, 3, 3})
    )
    << "Wrong block_shape";

    EXPECT_EQ(solver.output_name, "mpi.out") << "Wrong output_name";

    EXPECT_NEAR(solver.eps, 1e-10, 1e-11) << "Wrong eps";

    EXPECT_THAT(
        solver.l,
        ::testing::ElementsAreArray({1.0, 1.0, 1.0})
    ) 
    << "Wrong l";

    EXPECT_EQ(
        solver.boundaries,
        Lab07cpu::Boundaries({7.0, 7.0, 7.0, 7.0, 7.0, 7.0})
    )
    << "Wrong boundaries";

    EXPECT_EQ(solver.u_0, 0.0) << "Wrong u_0";
}

int get_rank()
{
    int rank;
    
    checkMPIErrors(MPI_Comm_rank(
        MPI_COMM_WORLD, 
        &rank
    ));

    return rank;
}

TEST (CpuInputTest, t2)
{
    if (get_rank() != 0)
        return;

    Lab07cpu solver(argc_, argv_);
    
    EXPECT_THAT(
        solver.process_grid_shape,
        ::testing::ElementsAreArray({1, 1, 2})
    )
    << "Wrong process_shape_grid";

    EXPECT_THAT(
        solver.block_shape,
        ::testing::ElementsAreArray({3, 3, 3})
    )
    << "Wrong block_shape";

    EXPECT_EQ(solver.output_name, "mpi.out") << "Wrong output_name";

    EXPECT_NEAR(solver.eps, 1e-10, 1e-11) << "Wrong eps";

    EXPECT_THAT(
        solver.l,
        ::testing::ElementsAreArray({1.0, 1.0, 2.0})
    ) 
    << "Wrong l";

    EXPECT_EQ(
        solver.boundaries,
        Lab07cpu::Boundaries({7.0, 0.0, 5.0, 0.0, 3.0, 0.0})
    )
    << "Wrong boundaries";

    EXPECT_EQ(solver.u_0, 5.0) << "Wrong u_0";
}

TEST (CpuOutputTest, t1)
{
    Lab07cpu solver(argc_, argv_);

    solver.solve();
    solver.write_answer();

    int cmp_result = system("cmp t1.out mpi.out");

    EXPECT_EQ(cmp_result, 0);
}

TEST (CpuOutputTest, t2)
{
    if (get_rank() != 0)
        return;

    Lab07cpu solver(argc_, argv_);

    solver.solve();
    solver.write_answer();

    int cmp_result = system("cmp t2.out mpi.out");

    EXPECT_EQ(cmp_result, 0);
}

TEST (FinalTest, single)
{
    Lab07::init(argc_, argv_);

    std::chrono::steady_clock::time_point dummy_start = std::chrono::steady_clock::now();

    // single CPU evaluation
    
    if (get_rank() == 0)
    {
        std::chrono::steady_clock::time_point cpu_start = std::chrono::steady_clock::now();
        
        Lab07cpu cpu_solver(argc_, argv_);

        cpu_solver.solve();

        cpu_solver.write_answer();

        std::chrono::steady_clock::time_point cpu_end = std::chrono::steady_clock::now();

        std::cout << "CPU time: " 
                  << std::chrono::duration_cast<std::chrono::seconds>(cpu_end - cpu_start).count() 
                  << "sec" 
                  << std::endl;
    }

    checkMPIErrors(MPI_Barrier(
        MPI_COMM_WORLD
    ));

    // MPI evaluation
    std::chrono::steady_clock::time_point mpi_start = std::chrono::steady_clock::now();

    Lab07 mpi_solver(argc_, argv_);

    mpi_solver.solve();

    mpi_solver.write_answer();

    std::chrono::steady_clock::time_point mpi_end = std::chrono::steady_clock::now();
    
    double start_d = std::chrono::duration_cast<std::chrono::seconds>(mpi_start - dummy_start).count(),
           end_d = std::chrono::duration_cast<std::chrono::seconds>(mpi_end - dummy_start).count();

    double min_start, max_end;

    checkMPIErrors(MPI_Reduce(
        &start_d,
        &min_start,
        1,
        MPI_DOUBLE,
        MPI_MIN,
        0,
        MPI_COMM_WORLD
    ));

    checkMPIErrors(MPI_Reduce(
        &end_d,
        &max_end,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    ));

    if (get_rank() == 0)
        std::cout << "MPI time: " << max_end - min_start << "sec" << std::endl;

    // check results
    
    if (get_rank() == 0)
    {
        int cmp_result = system("cmp mpi.out cpu.out");

        EXPECT_EQ(cmp_result, 0);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    
    argc_ = argc;
    argv_ = argv;

    int error_code;
    
    try
    {
        error_code = RUN_ALL_TESTS();
    }
    catch (std::exception &err)
    {
        std::cerr << "ERROR: \n" << err.what() << std::endl;
        Lab07::finalize();   
        return 1;
    }
    
    Lab07::finalize();

    return error_code;
}
