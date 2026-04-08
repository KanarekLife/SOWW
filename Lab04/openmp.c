#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "numgen.c"
#include "stdbool.h"

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
  unsigned long int *numbers = (unsigned long int*)malloc(inputArgument * sizeof(unsigned long int));
  numgen(inputArgument, numbers);

  struct timeval ins__tstart, ins__tstop;
  gettimeofday(&ins__tstart, NULL);
  
  // run your computations here (including OpenMP stuff)

  long twin_primes = 0;

  #pragma omp parallel for reduction(+:twin_primes)
  for (long i = 0; i < inputArgument - 1; i++) {
    if (is_prime(i) && is_prime(i+2)) {
      twin_primes++;
    }
  }

  // synchronize/finalize your computations
  gettimeofday(&ins__tstop, NULL);
  ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);

  printf("result=%ld\n", twin_primes);

}
