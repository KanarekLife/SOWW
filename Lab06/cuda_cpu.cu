#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "numgen.c"

const int THREADS_IN_BLOCK = 1024;

__device__
bool is_prime(long num) {
  if (num == 0) return false;
  if (num == 1) return false;
  if (num == 2) return true;

  for (long i = 2; i * i <= num; i++) {
    if (num % i == 0) {
      return false;
    }
  }

  return true;
}

__host__
void errorexit(const char *s) {
  printf("\n%s", s);
  exit(EXIT_FAILURE);
}

__global__
void calculate_thread_counts(int *thread_counts, long upper_limit, long total_threads) {
  const long my_index = blockIdx.x * blockDim.x + threadIdx.x;
  int local_count = 0;

  for (long idx = my_index; idx < total_threads; idx += blockDim.x * gridDim.x) {
    const long checked_num = idx * 2 + 1;
    if (checked_num <= upper_limit) {
      local_count += is_prime(checked_num) && is_prime(checked_num + 2);
    }
  }

  thread_counts[my_index] = local_count;
}

int main(int argc, char **argv) {
  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  const long inputArgument = ins__args.arg;
  const long neededThreads = inputArgument / 2;

  const int blocksingrid = (neededThreads / THREADS_IN_BLOCK) +
                           (neededThreads % THREADS_IN_BLOCK == 0 ? 0 : 1);
  const long launched_threads = (long)blocksingrid * THREADS_IN_BLOCK;

  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);

  int *d_thread_counts = NULL;
  if (cudaSuccess != cudaMalloc((void **)&d_thread_counts, launched_threads * sizeof(int)))
    errorexit("Error allocating GPU memory");
  if (cudaSuccess != cudaMemset(d_thread_counts, 0, launched_threads * sizeof(int)))
    errorexit("Error initializing GPU memory");

  calculate_thread_counts<<<blocksingrid, THREADS_IN_BLOCK>>>(
      d_thread_counts, inputArgument, neededThreads);

  if (cudaSuccess != cudaGetLastError())
    errorexit("Kernel launch failed");

  if (cudaSuccess != cudaDeviceSynchronize())
    errorexit("Kernel execution failed");

  int *h_thread_counts = (int *)malloc(launched_threads * sizeof(int));
  if (h_thread_counts == NULL)
    errorexit("Error allocating host memory");

  if (cudaSuccess != cudaMemcpy(h_thread_counts, d_thread_counts, launched_threads * sizeof(int), cudaMemcpyDeviceToHost))
    errorexit("Error copying result");

  int hresult = 0;
  for (long i = 0; i < launched_threads; i++) {
    hresult += h_thread_counts[i];
  }

  free(h_thread_counts);
  cudaFree(d_thread_counts);

  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);

  printf("result: %d\n", hresult);
  return 0;
}
