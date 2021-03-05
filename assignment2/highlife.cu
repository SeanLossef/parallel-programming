#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>

typedef unsigned int uint;
typedef unsigned short ushort;

// Result from last compute of world.
unsigned char *d_resultData=NULL;

// Current state of world. 
unsigned char *d_data=NULL;

// Host copy of the world.
unsigned char *h_data=NULL;

// Current width of world.
size_t g_worldWidth=0;

/// Current height of world.
size_t g_worldHeight=0;

/// Current data length (product of width and height)
size_t g_dataLength=0;  // g_worldWidth * g_worldHeight

static inline void HL_initialiaze( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMalloc(&d_data, (g_dataLength * sizeof(unsigned char)));
    cudaMalloc(&d_resultData, (g_dataLength * sizeof(unsigned char)));
    h_data = (unsigned char *)calloc(g_dataLength, sizeof(unsigned char)); 
}

static inline void HL_initAllZeros( size_t worldWidth, size_t worldHeight )
{
    // calloc init's to all zeros
}

static inline void HL_initAllOnes( size_t worldWidth, size_t worldHeight )
{
    // set all rows of world to true
    for(int i = 0; i < g_dataLength; i++)
    {
        h_data[i] = 1;
    }
}

static inline void HL_initOnesInMiddle( size_t worldWidth, size_t worldHeight )
{
    // set first 1 rows of world to true
    for(int i = 10*g_worldWidth; i < 11*g_worldWidth; i++)
    {
        if( (i >= ( 10*g_worldWidth + 10)) && (i < (10*g_worldWidth + 20)))
        {
            h_data[i] = 1;
        }
    }
}

static inline void HL_initOnesAtCorners( size_t worldWidth, size_t worldHeight )
{
    h_data[0] = 1; // upper left
    h_data[worldWidth-1]=1; // upper right
    h_data[(worldHeight * (worldWidth-1))]=1; // lower left
    h_data[(worldHeight * (worldWidth-1)) + worldWidth-1]=1; // lower right
}

static inline void HL_initSpinnerAtCorner( size_t worldWidth, size_t worldHeight )
{
    h_data[0] = 1; // upper left
    h_data[1] = 1; // upper left +1
    h_data[worldWidth-1]=1; // upper right
}

static inline void HL_initReplicator( size_t worldWidth, size_t worldHeight )
{
    size_t x, y;
    x = worldWidth/2;
    y = worldHeight/2;
    
    h_data[x + y*worldWidth + 1] = 1; 
    h_data[x + y*worldWidth + 2] = 1;
    h_data[x + y*worldWidth + 3] = 1;
    h_data[x + (y+1)*worldWidth] = 1;
    h_data[x + (y+2)*worldWidth] = 1;
    h_data[x + (y+3)*worldWidth] = 1; 
}

static inline void HL_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight )
{
    HL_initialiaze( worldWidth, worldHeight );

    switch(pattern)
    {
        case 0:
            HL_initAllZeros( worldWidth, worldHeight );
            break;
            
        case 1:
            HL_initAllOnes( worldWidth, worldHeight );
            break;
            
        case 2:
            HL_initOnesInMiddle( worldWidth, worldHeight );
            break;
        
        case 3:
            HL_initOnesAtCorners( worldWidth, worldHeight );
            break;

        case 4:
            HL_initSpinnerAtCorner( worldWidth, worldHeight );
            break;

        case 5:
            HL_initReplicator( worldWidth, worldHeight );
            break;
        
        default:
            printf("Pattern %u has not been implemented \n", pattern);
            exit(-1);
    }
}

// swap the pointers of pA and pB.
static inline void HL_swap( unsigned char **pA, unsigned char **pB)
{
    unsigned char *temp = *pA;
    *pA = *pB;
    *pB = temp;
}

// number of alive cells at 3x3 grid (excluding center)
__device__ static inline unsigned int HL_countAliveCells(const unsigned char* data, 
                    size_t x0, 
                    size_t x1, 
                    size_t x2, 
                    size_t y0, 
                    size_t y1,
                    size_t y2) 
{
    return (uint)data[x0+y0] + (uint)data[x0+y1] + (uint)data[x0+y2] + (uint)data[x1+y0] + (uint)data[x1+y2] + (uint)data[x2+y0] + (uint)data[x2+y1] + (uint)data[x2+y2];
}

// Don't Modify this function or your submitty autograding will not work
static inline void HL_printWorld(size_t iteration)
{
    int i, j;

    printf("Print World - Iteration %zu \n", iteration);
    
    for( i = 0; i < g_worldHeight; i++)
    {
        printf("Row %2d: ", i);
        for( j = 0; j < g_worldWidth; j++)
        {
            printf("%u ", (unsigned int)h_data[(i*g_worldWidth) + j]);
        }
        printf("\n");
    }

    printf("\n\n");
}

// CUDA kernel
__global__ void HL_kernel(const unsigned char* d_data,
    unsigned int worldWidth,
    unsigned int worldHeight,
    unsigned char* d_resultData)
{
    unsigned int index = blockIdx.x *blockDim.x + threadIdx.x;

    size_t x = index % worldWidth;
    size_t y = (int)(index / worldWidth);

    // calculate positions around current square
    size_t y0 = ((y + worldHeight - 1) % worldHeight) * worldWidth;
    size_t y1 = y * worldWidth;
    size_t y2 = ((y + 1) % worldHeight) * worldWidth;
    size_t x1 = x;
    size_t x0 = (x1 + worldWidth - 1) % worldWidth;
    size_t x2 = (x1 + 1) % worldWidth;

    // count alive cells around current square
    uint count = HL_countAliveCells(d_data, x0, x1, x2, y0, y1, y2);
    
    // compute if d_resultsData[y1 + x] is 0 or 1
    if (d_data[(y * worldWidth) + x] == 1) {
        if (count == 2 || count == 3)
            d_resultData[(y * worldWidth) + x] = 1;
        else
            d_resultData[(y * worldWidth) + x] = 0;
    } else {
        if (count == 3 || count == 6)
            d_resultData[(y * worldWidth) + x] = 1;
        else
            d_resultData[(y * worldWidth) + x] = 0;
    }
}

// Launch the kernel for a number of iterations
bool HL_kernelLaunch(unsigned char** d_data,
    unsigned char** d_resultData,
    size_t worldWidth,
    size_t worldHeight,
    size_t iterationsCount,
    ushort threadsCount)
{
    dim3 threadsPerBlock(threadsCount);
    dim3 blocksPerGrid(g_dataLength / threadsCount);

    for (int i = 0; i < iterationsCount; i++) {
        HL_kernel<<<blocksPerGrid, threadsPerBlock>>>(*d_data, worldWidth, worldHeight, *d_resultData);

        cudaDeviceSynchronize();

        HL_swap(d_resultData, d_data);
    }

    cudaDeviceSynchronize();

    return true;
}

int main(int argc, char *argv[])
{
    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int iterations = 0;
    unsigned int blocksize = 0;

    printf("This is the HighLife running in parallel on a GPU.\n");

    if( argc != 5 )
    {
        printf("HighLife requires 3 arguments, 1st is pattern number, 2nd the sq size of the world and 3rd is the number of itterations, 4th is the blocksize e.g. ./highlife 0 32 2 8 \n");
        exit(-1);
    }

    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    iterations = atoi(argv[3]);
    blocksize = atoi(argv[4]);
    
    HL_initMaster(pattern, worldSize, worldSize);
    // printf("AFTER INIT IS............\n");
    // HL_printWorld(0);
    
    cudaMemcpy(d_data, h_data, g_dataLength, cudaMemcpyHostToDevice);

    HL_kernelLaunch(&d_data, &d_resultData, g_worldWidth, g_worldHeight, iterations, blocksize);

    memset(h_data, 0, g_dataLength);
    cudaMemcpy(h_data, d_data, g_dataLength, cudaMemcpyDeviceToHost);
    
    // printf("######################### FINAL WORLD IS ###############################\n");
    // HL_printWorld(iterations);

    free(h_data);
    cudaFree(d_data);
    cudaFree(d_resultData);
    
    return true;
}
