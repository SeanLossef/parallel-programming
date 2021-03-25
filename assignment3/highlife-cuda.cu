#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>

// Convenient Types
typedef unsigned int uint;
typedef unsigned short ushort;

// CUDA external functions
extern "C" {
    bool HL_kernelLaunch(ushort threadsCount, int myrank);
    unsigned char** HL_initMaster(unsigned int pattern, size_t worldWidth, size_t worldHeight, int myrank, int numranks);
    void HL_terminate();
    void HL_setRow(unsigned char* buffer, int row);
}

// Result from last compute of world.
unsigned char* d_resultData=NULL;

// Current state of world. 
unsigned char* d_data=NULL;

// Current width of world.
size_t g_worldWidth=0;

// Current height of world.
size_t g_worldHeight=0;

// Current data length (product of width and height+2)
size_t g_dataLength=0;

// Current array length (product of width and height)
size_t g_arrayLength=0;


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


static inline void HL_initialiaze( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * (g_worldHeight + 2);
    g_arrayLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&d_data, (g_dataLength * sizeof(unsigned char)));
    cudaMallocManaged(&d_resultData, (g_dataLength * sizeof(unsigned char)));
}

static inline void HL_initAllZeros( size_t worldWidth, size_t worldHeight )
{
    // calloc init's to all zeros
}

static inline void HL_initAllOnes( size_t worldWidth, size_t worldHeight )
{
    // set all rows of world to true
    for(int i = g_worldWidth; i < g_dataLength - g_worldWidth; i++)
    {
        d_data[i] = 1;
    }
}

static inline void HL_initOnesInMiddle( size_t worldWidth, size_t worldHeight )
{
    // set first 1 rows of world to true
    for(int i = 10*g_worldWidth; i < 11*g_worldWidth; i++)
    {
        if( (i >= ( 10*g_worldWidth + 10)) && (i < (10*g_worldWidth + 20)))
        {
            d_data[i] = 1;
        }
    }
}

static inline void HL_initOnesAtCorners( size_t worldWidth, size_t worldHeight, int myrank, int numranks )
{
    if (myrank == 0) {
        d_data[worldWidth] = 1; // upper left
        d_data[worldWidth + worldWidth - 1] = 1; // upper right
    }

    if (myrank == numranks - 1) {
        d_data[worldWidth * worldHeight] = 1; // lower left
        d_data[(worldWidth * (worldHeight + 1)) - 1 ] = 1; // lower right
    }
}

static inline void HL_initSpinnerAtCorner( size_t worldWidth, size_t worldHeight, int myrank, int numranks )
{
    if (myrank == 0) {
        d_data[worldWidth] = 1; // upper left
        d_data[worldWidth + 1] = 1; // upper left +1
        d_data[worldWidth + worldWidth - 1] = 1; // upper right
    }
}

static inline void HL_initReplicator( size_t worldWidth, size_t worldHeight, int myrank, int numranks )
{
    if (myrank == numranks / 2) {
        size_t x, y;
        x = worldWidth/2;
        y = 1;
        
        d_data[x + y*worldWidth + 1] = 1; 
        d_data[x + y*worldWidth + 2] = 1;
        d_data[x + y*worldWidth + 3] = 1;
        d_data[x + (y+1)*worldWidth] = 1;
        d_data[x + (y+2)*worldWidth] = 1;
        d_data[x + (y+3)*worldWidth] = 1;
    }
}

unsigned char** HL_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight, int myrank, int numranks )
{
    int cudaDeviceCount = -1;
    cudaError_t cE = cudaSuccess;

    if ((cE = cudaGetDeviceCount(&cudaDeviceCount)) != cudaSuccess) {
        printf("Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount);
        exit(-1);
    }
    if ((cE = cudaSetDevice(myrank % cudaDeviceCount)) != cudaSuccess) {
        printf("Unable to have rank %d set to cuda device %d, error is %d\n", myrank, (myrank % cudaDeviceCount), cE);
        exit(-1);
    }
    
    HL_initialiaze(worldWidth, worldHeight);

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
            HL_initOnesAtCorners( worldWidth, worldHeight, myrank, numranks );
            break;

        case 4:
            HL_initSpinnerAtCorner( worldWidth, worldHeight, myrank, numranks );
            break;

        case 5:
            HL_initReplicator( worldWidth, worldHeight, myrank, numranks );
            break;
        
        default:
            printf("Pattern %u has not been implemented \n", pattern);
            exit(-1);
    }

    return &d_data;
}

// swap the pointers of pA and pB.
static inline void HL_swap( unsigned char **pA, unsigned char **pB)
{
    unsigned char *temp = *pA;
    *pA = *pB;
    *pB = temp;
}

// Set a specific row of data
void HL_setRow(unsigned char* buffer, int row) {
    for (int i = 0; i < g_worldWidth; i++) {
        d_data[i + (row * g_worldWidth)] = buffer[i];
    }
}

// Free CUDA memory
void HL_terminate() {
    cudaFree(d_data);
    cudaFree(d_resultData);
}


// CUDA kernel
__global__ void HL_kernel(unsigned char* d_data, unsigned int worldWidth, unsigned int worldHeight, unsigned char* d_resultData) {
    unsigned int index = blockIdx.x *blockDim.x + threadIdx.x;

    size_t x = index % worldWidth;
    size_t y = (int)(index / worldWidth) + 1;

    // calculate positions around current square
    size_t y0 = ((y + (worldHeight + 2) - 1) % (worldHeight + 2)) * worldWidth;
    size_t y1 = y * worldWidth;
    size_t y2 = ((y + 1) % (worldHeight + 2)) * worldWidth;
    size_t x1 = x;
    size_t x0 = (x1 + worldWidth - 1) % worldWidth;
    size_t x2 = (x1 + 1) % worldWidth;

    // count alive cells around current square
    uint count = HL_countAliveCells(d_data, x0, x1, x2, y0, y1, y2);
    
    // compute if d_resultData[y1 + x] is 0 or 1
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

    __syncthreads();
}

// Launch the kernel for a number of iterations
bool HL_kernelLaunch(ushort threadsCount, int myrank)
{
    dim3 threadsPerBlock(threadsCount);
    dim3 blocksPerGrid(g_arrayLength / threadsCount);

    HL_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, g_worldWidth, g_worldHeight, d_resultData);

    cudaDeviceSynchronize();

    HL_swap(&d_resultData, &d_data);

    // Clear ghost rows
    for (int i = 0; i < g_worldWidth; i++) {
        d_data[i] = 0;
        d_data[(g_worldWidth * (g_worldHeight + 1)) + i] = 0;
    }

    cudaDeviceSynchronize();

    return true;
}
