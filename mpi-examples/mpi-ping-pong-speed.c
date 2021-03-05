#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int PING_PONG_LIMIT = 32768;

int main(int argc, char** argv) 
{

  // Initialize the MPI environment
  MPI_Init(NULL, NULL);
  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // We are assuming 2 processes for this task
  if (world_size != 2) 
    {
    fprintf(stderr, "World size must be two for %s\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
    }

  int ping_pong_count = 0;
  int partner_rank = (world_rank + 1) % 2;

  double start_time = MPI_Wtime();
  while (ping_pong_count < PING_PONG_LIMIT) 
    {
      if (world_rank == ping_pong_count % 2) 
	{
	  // Increment the ping pong count before you send it
	  ping_pong_count++;
	  MPI_Send(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
	  // printf("%d sent and incremented ping_pong_count %d to %d\n",  world_rank, ping_pong_count, partner_rank);
	} 
      else 
	{
	  MPI_Recv(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD,
		   MPI_STATUS_IGNORE);
	  // printf("%d received ping_pong_count %d from %d\n", world_rank, ping_pong_count, partner_rank);
	}
  }
  MPI_Barrier( MPI_COMM_WORLD );
  double end_time = MPI_Wtime();
  double time_per_message = (end_time - start_time ) / (double)PING_PONG_LIMIT;

  if( world_rank == 0 )  
  printf("Rank %d: Sent and Recv'd %d messages of size %ld in %lf seconds or %lf avg microseconds per message \n", 
	 world_rank, PING_PONG_LIMIT, sizeof(MPI_INT), (end_time - start_time), (time_per_message * 1000000.0));

  MPI_Finalize();
}