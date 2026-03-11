#include "utility.h"
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

const int MASTER_RANK = 0;

const int TAG_SCHEDULEJOB = 0;
const int TAG_DATA = 1;
const int TAG_STOP = 2;
const int TAG_DONE = 3;

typedef struct {
  long start;
  long end;
} JobRequest;

typedef struct {
  long result;
  JobRequest request;
  bool sent;
} JobBuffer;

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

bool get_next_request(long end, long *current_start, JobRequest* request) {
  const int REQUEST_RANGE_SIZE = 1000;

  if (*current_start > end) {
    return false;
  }

  request->start = *current_start;
  request->end = *current_start + REQUEST_RANGE_SIZE;
  *current_start += REQUEST_RANGE_SIZE + 1;

  if (request->end > end) {request->end = end;
    request->end = end;
  }

  return true;
}

int schedule_job(JobBuffer *buffers, MPI_Request *mpi_requests, int rank, JobRequest request) {
  buffers[rank-1].request.start = request.start;
  buffers[rank-1].request.end = request.end;
  buffers[rank-1].sent = true;
  return MPI_Isend(&buffers[rank-1].request, 2, MPI_LONG, rank, TAG_SCHEDULEJOB, MPI_COMM_WORLD, &mpi_requests[rank-1]);
}

int schedule_stop(JobBuffer *buffers, MPI_Request *mpi_requests, int rank) {
  buffers[rank-1].sent = true;
  return MPI_Isend(NULL, 0, MPI_LONG, rank, TAG_STOP, MPI_COMM_WORLD, &mpi_requests[rank-1]);
}

int receive_result(JobBuffer *buffers, MPI_Request *mpi_requests, int rank) {
  buffers[rank-1].sent = false;
  return MPI_Irecv(&buffers[rank-1].result, 1, MPI_LONG, rank, MPI_ANY_TAG, MPI_COMM_WORLD, &mpi_requests[rank-1]);
}

long master(int nproc, long start, long end) {
  const int slaves_count = nproc - 1;
  int alive_slaves_count = slaves_count;
  long current_start = start;
  long result = 0;

  JobBuffer *buffers = malloc(sizeof(JobBuffer) * slaves_count);
  if (buffers == NULL) {
    fprintf(stderr, "Failed to allocate memory for job buffers\n");
    exit(EXIT_FAILURE);
  }

  MPI_Request *mpi_requests = malloc(sizeof(MPI_Request) * slaves_count);
  if (mpi_requests == NULL) {
    fprintf(stderr, "Failed to allocate memory for MPI requests\n");
    free(buffers);
    exit(EXIT_FAILURE);
  }

  JobRequest request;

  for (int i = 1; i <= slaves_count; ++i) {
    receive_result(buffers, mpi_requests, i);
  }

  while (alive_slaves_count > 0) {
    int slave_rank = 0;
    int request_done = false;
    MPI_Status mpi_status;

    if (current_start > end) {
      MPI_Waitany(slaves_count, mpi_requests, &slave_rank, &mpi_status);
      request_done = true;
    } else {
      MPI_Testany(slaves_count, mpi_requests, &slave_rank, &request_done, &mpi_status);
    }
    slave_rank++;

    if (buffers[slave_rank-1].sent) {
        receive_result(buffers, mpi_requests, slave_rank);
        continue;
    }

    if (mpi_status.MPI_TAG == TAG_DONE) {
      alive_slaves_count--;
      continue;
    }

    if (mpi_status.MPI_TAG != TAG_DATA) {
      continue;
    }

    result += buffers[slave_rank-1].result;

    JobRequest request;
      
    if (get_next_request(end, &current_start, &request)) {
      schedule_job(buffers, mpi_requests, slave_rank, request);
    } else {
      schedule_stop(buffers, mpi_requests, slave_rank);
    }
  }

  free(buffers);
  free(mpi_requests);

  return result;
}

void slave(long hard_limit) {
  MPI_Status recv_status, send_status;
  MPI_Request recv_request, send_request;

  JobRequest request;
  long result = 0;
  
  JobRequest request_buffer;
  long result_buffer = 0;

  MPI_Send(&result_buffer, 1, MPI_LONG, MASTER_RANK, TAG_DATA, MPI_COMM_WORLD);
  MPI_Irecv(&request_buffer, 2, MPI_LONG, MASTER_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);
  MPI_Isend(&result_buffer, 1, MPI_LONG, MASTER_RANK, TAG_DATA, MPI_COMM_WORLD, &send_request);

  while (true) {
    MPI_Wait(&recv_request, &recv_status);
    request = request_buffer;
    MPI_Irecv(&request_buffer, 2, MPI_LONG, MASTER_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);
    
    if (recv_status.MPI_TAG == TAG_STOP) {
      break;
    }

    if (recv_status.MPI_TAG != TAG_SCHEDULEJOB) {
      continue;
    }

    result = count_twin_primes_in_range(request.start, request.end, hard_limit);
    MPI_Wait(&send_request, &send_status);
    
    result_buffer = result;
    MPI_Isend(&result_buffer, 1, MPI_LONG, MASTER_RANK, TAG_DATA, MPI_COMM_WORLD, &send_request);
  }

  MPI_Wait(&send_request, &send_status);
  MPI_Send(NULL, 0, MPI_LONG, MASTER_RANK, TAG_DONE, MPI_COMM_WORLD);
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
