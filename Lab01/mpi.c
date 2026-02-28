#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>
#include <mpi.h>

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

  //program input argument
  long inputArgument = ins__args.arg; 

  struct timeval ins__tstart, ins__tstop;

  int myrank,nproc;

  long total;
  
  MPI_Init(&argc,&argv);

  // obtain my rank
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  // and the number of processes
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);

  long total_local = 0;
  long partition_size = inputArgument / nproc;
  long from = partition_size * myrank;
  long to = from + partition_size;

  if(!myrank)
      gettimeofday(&ins__tstart, NULL);


  // run your computations here (including MPI communication)
  // printf("\nProcess %d", myrank);
  // fflush(stdout);

  for (; from < to; from++) {
    if (is_prime(from)) {
      total_local++;
    }
  }

  printf("parition: %d -> %ld\n", myrank, total_local);
  fflush(stdout);

  MPI_Reduce(&total_local, &total, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);


  // synchronize/finalize your computations

  if (!myrank) {
    gettimeofday(&ins__tstop, NULL);
    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);
    printf("result=%ld\n", total);
  }
  
  MPI_Finalize();

}
