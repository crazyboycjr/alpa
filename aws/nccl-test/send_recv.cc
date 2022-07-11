#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#include <vector>
#include <chrono>

#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"


#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}


enum class TestType {
  Allreduce,
  SendRecv,
};


int main(int argc, char* argv[])
{
  TestType test_type = TestType::SendRecv;
  int count = 32*1024*1024;
  int warmup = 10;
  int num_iters = 100;

  int myRank, nRanks, localRank = 0;

  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  printf("My rank: %d\n", myRank);
  if (myRank == 0) {
    printf("World count: %d\n", nRanks);
  }


  //calculating localRank which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  printf("My local rank: %d\n", localRank);


  //each process is using two GPUs
  int nDev = 1;
  printf("Number of devices to use: %d\n", nDev);

  if (myRank == 0) {
    setenv("CUDA_VISIBLE_DEVICES", "0", 1);
  } else {
    setenv("CUDA_VISIBLE_DEVICES", "7", 1);
  }

  auto sendbuff = std::vector<float*>(nDev);
  auto recvbuff = std::vector<float*>(nDev);
  auto s = std::vector<cudaStream_t>(nDev);

  //picking GPUs based on localRank
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(localRank*nDev + i));
    CUDACHECK(cudaMallocHost(&sendbuff[i], count * sizeof(float)));
    CUDACHECK(cudaMallocHost(&recvbuff[i], count * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, count * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, count * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s[i]));
  }


  ncclUniqueId id;
  ncclComm_t comms[nDev]; // now you are willing to use nDev.....???


  //generating NCCL unique ID at one process and broadcasting it to all
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));


  //initializing NCCL, group API is required around ncclCommInitRank as it is
  //called across multiple GPUs in each thread/process
  NCCLCHECK(ncclGroupStart());
  for (int i=0; i<nDev; i++) {
     CUDACHECK(cudaSetDevice(localRank*nDev + i));
     NCCLCHECK(ncclCommInitRank(comms+i, nRanks*nDev, id, myRank*nDev + i));
  }
  NCCLCHECK(ncclGroupEnd());


  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread/process

  auto do_allreduce = [&]() {
    NCCLCHECK(ncclGroupStart());
    for (int i=0; i<nDev; i++) {
      NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], count, ncclFloat, ncclSum,
                comms[i], s[i]));
    }
    NCCLCHECK(ncclGroupEnd());
  };

  auto do_send_or_recv = [&]() {
    NCCLCHECK(ncclGroupStart());
    for (int i=0; i<nDev; i++) {
      auto peer = (myRank + nRanks / 2) % nRanks;
      if (myRank < nRanks / 2) {
        NCCLCHECK(ncclSend(sendbuff[i], count, ncclInt, peer*nDev + i, comms[i], s[i]));
      } else {
        NCCLCHECK(ncclRecv(recvbuff[i], count, ncclInt, peer*nDev + i, comms[i], s[i]));
      }
    }
    NCCLCHECK(ncclGroupEnd());
  };

  auto sync = [&]() {
    for (int i=0; i<nDev; i++) {
      CUDACHECK(cudaStreamSynchronize(s[i]));
    }
  };

  auto run_test_once = [&]() {
    switch (test_type) {
      case TestType::Allreduce: {
        do_allreduce();
        break;
      }
      case TestType::SendRecv: {
        do_send_or_recv();
        break;
      }
      default: {
        fprintf(stderr, "Unknown test %d", static_cast<int>(test_type));
        assert(0);
      }
    }
  };

  printf("start warmup for %d iterations.\n", warmup);

  for (auto i = 0; i < warmup; i++) {
    run_test_once();
  }

  sync();

  // benchmark starts here
  printf("start benchmark for %d iterations.\n", num_iters);

  auto start = std::chrono::high_resolution_clock::now();

  for (auto i = 0; i < num_iters; i++) {
    run_test_once();
  }

  sync();

  //synchronizing on CUDA stream to complete NCCL communication
  auto end = std::chrono::high_resolution_clock::now();

  auto total_dura_ns = (end - start).count();
  double coeff = test_type == TestType::Allreduce ? (2.0 * (nRanks - 1) / nRanks) : 1.0;

  printf("total duration: %.3f us, num_iters: %d, avg: %.3f us, speed: %f Gbps \n",
          total_dura_ns / 1e3, num_iters, total_dura_ns / 1e3 / num_iters,
          coeff * nDev * count * sizeof(float) * num_iters * 8.0 / (total_dura_ns / 1e9) / 1e9);


  //freeing device memory
  for (int i=0; i<nDev; i++) {
     CUDACHECK(cudaFreeHost(sendbuff[i]));
     CUDACHECK(cudaFreeHost(recvbuff[i]));
  }


  //finalizing NCCL
  for (int i=0; i<nDev; i++) {
     ncclCommDestroy(comms[i]);
  }


  //finalizing MPI
  MPICHECK(MPI_Finalize());


  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}