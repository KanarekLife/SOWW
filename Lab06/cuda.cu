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

  for (long i = 2; i*i <= num; i++) {
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
void calculate(int *result, long upper_limit) {
  __shared__ int shared[THREADS_IN_BLOCK];

  const int thread_id = threadIdx.x;
  const long my_index = blockIdx.x * blockDim.x + threadIdx.x;
  const long checked_num = my_index * 2 + 1;

  if (checked_num > upper_limit) {
    shared[thread_id] = 0;
  } else {
    shared[thread_id] = is_prime(checked_num) && is_prime(checked_num + 2);
  }

  __syncthreads();

  for(int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (thread_id < s) {
      shared[thread_id] += shared[thread_id + s];
    }
    __syncthreads();
  }

  if (thread_id == 0) {
    atomicAdd(result, shared[0]);
  }
}


int main(int argc,char **argv) {

  Args ins__args;
  parseArgs(&ins__args, &argc, argv);
  
  //program input argument
  const long inputArgument = ins__args.arg;
  const long neededThreads = inputArgument / 2;
  
  int blocksingrid = (neededThreads / THREADS_IN_BLOCK) + (neededThreads % THREADS_IN_BLOCK == 0 ? 0 : 1);

  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);
  
  // device result
  int *dresult;
  if (cudaSuccess != cudaMalloc((void**)&dresult, sizeof(int)))
      errorexit("Error allocating GPU memory");
  if (cudaSuccess != cudaMemset(dresult, 0, sizeof(int)))
      errorexit("Error initializing GPU memory");

  // run your CUDA kernel(s) here

  calculate<<<blocksingrid, THREADS_IN_BLOCK>>>(dresult, inputArgument);

  // synchronize/finalize your CUDA computations

  if (cudaSuccess != cudaGetLastError())
      errorexit("Kernel launch failed");

  if (cudaSuccess != cudaDeviceSynchronize())
      errorexit("Kernel execution failed");

  int hresult = 0;
  if (cudaSuccess != cudaMemcpy(&hresult, dresult, sizeof(int), cudaMemcpyDeviceToHost))
      errorexit("Error copying result");

  // free GPU memory
  cudaFree(dresult);

  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
  
  printf("result: %d\n", hresult);
}
