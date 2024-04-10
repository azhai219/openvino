// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include "oneapi/ccl.hpp"
#include <mpi.h>
#include <vector>
#include <iostream>

inline void mpi_finalize() {
    int is_finalized = 0;
    MPI_Finalized(&is_finalized);

    if (!is_finalized)
        MPI_Finalize();
}

TEST(CCLTest, AllGatherTest) {
        const size_t count = 128;

    size_t i = 0;
    std::vector<int> send_buf;
    std::vector<int> recv_buf;
    std::vector<size_t> recv_counts;

    ccl::init();

    int size, rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    auto comm = ccl::create_communicator(size, rank, kvs);

    rank = comm.rank();
    size = comm.size();

    send_buf.resize(count, rank);
    recv_buf.resize(size * count);
    recv_counts.resize(size, count);

    /* modify send_buf */
    for (i = 0; i < count; i++) {
        send_buf[i] += 1;
    }

    /* invoke allgatherv */
    ccl::allgatherv(send_buf.data(), count, recv_buf.data(), recv_counts, comm).wait();

    /* check correctness of recv_buf */
    for (i = 0; i < count; i++) {
        if (recv_buf[i] != rank + 1) {
            recv_buf[i] = -1;
        }
    }

    /* print out the result of the test */
    if (rank == 0) {
        for (i = 0; i < count; i++) {
            ASSERT_NE(recv_buf[i], -1);
        }
        ASSERT_EQ(i, count);
    }
}