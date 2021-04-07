#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

/* 
 * 64 bit, free running clock for POWER9/AiMOS system
 *  Has 512MHz resolution.
 */
unsigned long long aimos_clock_read(void)
{
    unsigned int tbl, tbu0, tbu1;

    do {
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);

    return (((unsigned long long)tbu0) << 32) | tbl;
}

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (num_ranks < 32) {
        printf("Not enough MPI ranks available for use.\n");
        exit(-1);
    }

    // Get the rank of the process
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Verify arguments
    if(argc != 2) {
        printf("Correct execution: ./assignment4-exe <filepath>\n");
        exit(-1);
    }

    // Fetch arguments
    char *filename = argv[1];

    // Run tests
    int i;
    int j;
    int k;
    for (i = 0; i < 6; i++) {
        num_ranks = 2 << i; // 2, 4, 8, 16, 32, 64

        for (j = 0; j < 8; j++) {
            MPI_File fh;
            MPI_Offset my_offset;
            MPI_Status status;
            unsigned long long t1, t2;
            uint block_size;
            int *buff;
            int x;

            block_size = (128 * 1024) << j; // 128K, 256K, 512K, 1M, 2M, 4M, 8M and 16M
            my_offset = block_size * my_rank * sizeof(unsigned char);

            buff = (int *)calloc(block_size, sizeof(unsigned char));

            //// WRITE TEST ////
            t1 = aimos_clock_read();

            // Do 32 write tests
            for (k = 0; k < 32; k++) {
                MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);

                //MPI_File_set_view(fh, 0, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, (char *)NULL, (MPI_Info)NULL);
                
                // Write to my section of the file -- Only implement as many ranks as needed for test
                if (my_rank < num_ranks) {
                    MPI_File_write_at(fh, my_offset, buff, block_size, MPI_UNSIGNED_CHAR, &status);
                }

                MPI_File_close(&fh);

                MPI_Barrier(MPI_COMM_WORLD);
            }

            // Print elapsed time
            if (my_rank == 0) {
                t2 = aimos_clock_read();
                printf("WRITE TEST COMPLETE: %d MPI RANKS -- %d BLOCK SIZE ====> ELAPSED TIME: %fs\n", num_ranks, block_size / 1024, ((float)(t2 - t1)) / 512000000);
            }

            //// READ TEST ////
            t1 = aimos_clock_read();

            // Do 32 read tests
            for (k = 0; k < 32; k++) {
                MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
                
                // Write to my section of the file -- Only implement as many ranks as needed for test
                if (my_rank < num_ranks) {
                    MPI_File_read_at(fh, my_offset, buff, block_size, MPI_UNSIGNED_CHAR, &status);
                }

                MPI_File_close(&fh);

                MPI_Barrier(MPI_COMM_WORLD);
            }

            // Print elapsed time
            if (my_rank == 0) {
                t2 = aimos_clock_read();
                printf("READ TEST COMPLETE:  %d MPI RANKS -- %d BLOCK SIZE ====> ELAPSED TIME: %fs\n", num_ranks, block_size / 1024, ((float)(t2 - t1)) / 512000000);
            }

            free(buff);
        }
    }

    // Finalize the MPI environment
    MPI_Finalize();
}