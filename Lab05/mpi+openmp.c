#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include "numgen.c"
#include "stdbool.h"

const int TAG_RESULT = 0;

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

int main(int argc,char **argv) {

  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  //set number of threads
  omp_set_num_threads(ins__args.n_thr);
  
  //program input argument
  long inputArgument = ins__args.arg; 

  struct timeval ins__tstart, ins__tstop;

  int threadsupport;
  int myrank,nproc;
  unsigned long int *numbers;
  // Initialize MPI with desired support for multithreading -- state your desired support level

  MPI_Init_thread(&argc, &argv,MPI_THREAD_FUNNELED,&threadsupport); 

  if (threadsupport<MPI_THREAD_FUNNELED) {
    printf("\nThe implementation does not support MPI_THREAD_FUNNELED, it supports level %d\n",threadsupport);
    MPI_Finalize();
    return -1;
  }
  
  // obtain my rank
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  // and the number of processes
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);

  if(!myrank){
      gettimeofday(&ins__tstart, NULL);
	numbers = (unsigned long int*)malloc(inputArgument * sizeof(unsigned long int));
  	numgen(inputArgument, numbers);
  }
  // run your computations here (including MPI communication and OpenMP stuff)

  long start = (myrank * 2) + 1;
  long step = nproc * 2;

  long twin_primes = 0;

  #pragma omp parallel for reduction(+:twin_primes)
  for (long i = start; i <= inputArgument - 2; i += step) {
    if (is_prime(i) && is_prime(i + 2)) {
      twin_primes++;
    }
  }
  
  MPI_Send(&twin_primes, 1, MPI_LONG, 0, TAG_RESULT, MPI_COMM_WORLD);

  // synchronize/finalize your computations

  long total_twin_primes = 0;
  MPI_Status status;

  if (!myrank) {
    long temp;
    for (int i = 0; i < nproc; i++) {
      MPI_Recv(&temp, 1, MPI_LONG, i, TAG_RESULT, MPI_COMM_WORLD, &status);
      total_twin_primes += temp;
    }

    gettimeofday(&ins__tstop, NULL);
    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
    printf("result=%ld\n", total_twin_primes);
  }
    
  MPI_Finalize();
  
}
