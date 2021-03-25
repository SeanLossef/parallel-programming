#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

// Convenient Types
typedef unsigned int uint;
typedef unsigned short ushort;

// CUDA external functions
extern bool HL_kernelLaunch(ushort threadsCount, int myrank);
extern unsigned char** HL_initMaster(unsigned int pattern, size_t worldWidth, size_t worldHeight, int myrank, int numranks);
extern void HL_terminate();
extern void HL_setRow(unsigned char* buffer, int row);


// Current state of world. 
unsigned char **c_d_data=NULL;

// Current width of world.
size_t c_g_worldWidth=0;

/// Current height of world.
size_t c_g_worldHeight=0;

/// Current data length (product of width and height)
size_t c_g_dataLength=0;  // c_g_worldWidth * c_g_worldHeight


// Print the world
void printWorld(unsigned char *buffer, int rank)
{
    int i, j;
    
    for( i = 0; i < c_g_worldHeight; i++)
    {
        printf("Row %2lu: ", i + (rank * c_g_worldHeight) + 1);
        for( j = 0; j < c_g_worldWidth; j++)
        {
            printf("%u ", (unsigned int)buffer[(i*c_g_worldWidth) + j]);
        }
        printf("\n");
    }
}


int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Fetch Highlife arguments
    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int iterations = 0;
    unsigned int blocksize = 0;
    unsigned int output = 0;

    // Verify arguments
    if( argc != 6 ) {
        printf("HighLife requires 5 arguments, 1st is pattern number, 2nd the sq size of the world and 3rd is the number of itterations, 4th is the blocksize, 5th is output enable (0/1) e.g. ./highlife 0 32 2 8 1 \n");
        exit(-1);
    }

    // Fetch arguments
    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    iterations = atoi(argv[3]);
    blocksize = atoi(argv[4]);
    output = atoi(argv[5]);

    // Calculate world sizes
    c_g_worldWidth = worldSize;
    c_g_worldHeight = worldSize / num_ranks;
    c_g_dataLength = c_g_worldWidth * c_g_worldHeight;

    // Initialize data structures
    c_d_data = HL_initMaster(pattern, c_g_worldWidth, c_g_worldHeight, world_rank, num_ranks);

    MPI_Request request[4];
    MPI_Status status[4];

    double t1, t2;
    t1 = MPI_Wtime();

    // Iterate
    unsigned char *sendbuff1 = malloc(worldSize * sizeof(unsigned char));
    unsigned char *sendbuff2 = malloc(worldSize * sizeof(unsigned char));
    unsigned char *recvbuff1 = malloc(worldSize * sizeof(unsigned char));
    unsigned char *recvbuff2 = malloc(worldSize * sizeof(unsigned char));

    for (int i = 0; i < iterations; i++) {
        MPI_Irecv(recvbuff1, worldSize, MPI_UNSIGNED_CHAR, (num_ranks + world_rank - 1) % num_ranks, 0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(recvbuff2, worldSize, MPI_UNSIGNED_CHAR, (world_rank + 1) % num_ranks, 1, MPI_COMM_WORLD, &request[1]);

        for (int j = 0; j < worldSize; j++) {
            sendbuff1[j] = (*c_d_data)[j + worldSize];
            sendbuff2[j] = (*c_d_data)[j + (worldSize * c_g_worldHeight)];
        }

        MPI_Isend(sendbuff1, worldSize, MPI_UNSIGNED_CHAR, (num_ranks + world_rank - 1) % num_ranks, 1, MPI_COMM_WORLD, &request[2]);
        MPI_Isend(sendbuff2, worldSize, MPI_UNSIGNED_CHAR, (world_rank + 1) % num_ranks, 0, MPI_COMM_WORLD, &request[3]);

        MPI_Waitall(4, request, status);

        HL_setRow(recvbuff1, 0);
        HL_setRow(recvbuff2, c_g_worldHeight+1);

        HL_kernelLaunch(blocksize, world_rank);
    }

    // Wait for all ranks to complete
    MPI_Barrier(MPI_COMM_WORLD);

    // Calculate time taken
    t2 = MPI_Wtime();
    if (world_rank == 0)
        printf("Elapsed Time: %fs\n\n", t2-t1);

    // Print world if output flag set
    if (output) {
        if (world_rank == 0) {
            printf("Print World\n");
            printWorld((*c_d_data) + c_g_worldWidth, 0);

            unsigned char *buffer;

            for (int i = 1; i < num_ranks; i++) {
                buffer = (unsigned char *) malloc((c_g_dataLength) * sizeof(unsigned char));

                MPI_Recv(buffer, c_g_dataLength, MPI_UNSIGNED_CHAR, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                printWorld(buffer, i);

                free(buffer);
            }

            printf("\n\n");
        } else {
            MPI_Send(&(*c_d_data)[worldSize], c_g_dataLength, MPI_UNSIGNED_CHAR, 0, 2, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Free memory
    HL_terminate();
    free(sendbuff1);
    free(sendbuff2);
    free(recvbuff1);
    free(recvbuff2);

    // Finalize the MPI environment.
    MPI_Finalize();
}
