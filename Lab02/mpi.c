#include "utility.h"
#include <stdbool.h>
#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>

const int MASTER_RANK = 0;

const int TAG_SCHEDULEJOB = 0;
const int TAG_DATA = 1;
const int TAG_STOP = 2;

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

long count_twin_primes_in_range(long start, long end, long hard_limit) {
  if (start % 2 == 0) {
    start++;
  }

  if (end > hard_limit) {
    end = hard_limit;
  }

  long twin_primes = 0;
  bool was_prev_prime = false;

  for (long i = start; i <= end; i+=2) {
    if (is_prime(i) && was_prev_prime) {
      twin_primes++;
      was_prev_prime = true;
      continue;
    }
    
    if (is_prime(i)) {
      was_prev_prime = true;
      continue;
    }
    
    was_prev_prime = false;
  }

  if (!was_prev_prime) {
    return twin_primes;
  }

  long next_potential_prime = end + 1 + (end % 2 == 0 ? 0 : 1);

  if (next_potential_prime <= hard_limit && is_prime(next_potential_prime)) {
    return twin_primes + 1;
  }

  return twin_primes;
}

long master(int nproc, long start, long end) {
  const int RANGE_SIZE = 1000;

  MPI_Status recv_status;
  long result = 0;
  int slaves_count = nproc - 1;
  long curr_start = start;

  while (slaves_count > 0) {
    long partial_result;
    MPI_Recv(&partial_result, 1, MPI_LONG, MPI_ANY_SOURCE, TAG_DATA, MPI_COMM_WORLD, &recv_status);
    result += partial_result;

    if (curr_start > end) {
      MPI_Send(NULL, 0, MPI_LONG, recv_status.MPI_SOURCE, TAG_STOP, MPI_COMM_WORLD);
      slaves_count--;
      continue;
    }

    long request[2] = {curr_start, curr_start + RANGE_SIZE};
    curr_start += RANGE_SIZE + 1;
    if (request[1] > end) {
      request[1] = end;
    }

    MPI_Send(request, 2, MPI_LONG, recv_status.MPI_SOURCE, TAG_SCHEDULEJOB, MPI_COMM_WORLD);
  }

  return result;
}

void slave(long hard_limit) {
  MPI_Status recv_status;
  long result = 0;

  while (true) {
    MPI_Send(&result, 1, MPI_LONG, MASTER_RANK, TAG_DATA, MPI_COMM_WORLD);
    MPI_Probe(MASTER_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_status);

    if (recv_status.MPI_TAG == TAG_STOP) {
      break;
    }
    
    if (recv_status.MPI_TAG == TAG_SCHEDULEJOB) {
      long range_to_process[2];
      MPI_Recv(range_to_process, 2, MPI_LONG, MASTER_RANK, TAG_SCHEDULEJOB, MPI_COMM_WORLD, &recv_status);
      result = count_twin_primes_in_range(range_to_process[0], range_to_process[1], hard_limit);
      continue;
    }
  }
}

int main(int argc,char **argv) {
  Args ins__args;
  parseArgs(&ins__args, &argc, argv);

  const int INITIAL_NUMBER = ins__args.start; 
  const int FINAL_NUMBER = ins__args.stop;
  struct timeval ins__tstart, ins__tstop;

  int myrank,nproc;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);

  if (myrank == MASTER_RANK) {
    gettimeofday(&ins__tstart, NULL);
  }

  // run your computations here (including MPI communication)

  long twin_primes = 0;

  if (myrank == MASTER_RANK) {
    twin_primes = master(nproc, INITIAL_NUMBER, FINAL_NUMBER);
  } else {
    slave(FINAL_NUMBER);
  }

  // synchronize/finalize your computations

  if (myrank == MASTER_RANK) {
    gettimeofday(&ins__tstop, NULL);
    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);

    printf("result=%ld\n", twin_primes);
  }
  
  MPI_Finalize();
}
