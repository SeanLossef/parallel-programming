#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

extern "C" 
{
void runCudaLand( int myrank );
}

__global__ void Hello_kernel( int myrank );

void runCudaLand( int myrank )
{
  int cudaDeviceCount = -1;
  int assignedCudaDevice = -1;
  cudaError_t cE = cudaSuccess;

  if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
    {
      printf(" Unable to determine cuda device count, error is %d, count is %d\n", 
	     cE, cudaDeviceCount );
      exit(-1);
    }
  
  if( (cE = cudaSetDevice( myrank % cudaDeviceCount )) != cudaSuccess )
    {
      printf(" Unable to have rank %d set to cuda device %d, error is %d \n", 
	     myrank, (myrank % cudaDeviceCount), cE);
      exit(-1);
    }

  if( (cE = cudaGetDevice( &assignedCudaDevice )) != cudaSuccess )
    {
      printf(" Unable to have rank %d set to cuda device %d, error is %d \n", 
	     myrank, (myrank % cudaDeviceCount), cE);
      exit(-1);
    }

  if( assignedCudaDevice != (myrank % cudaDeviceCount) )
    {
      printf("MPI Rank %d: assignedCudaDevice %d NOT EQ to (myrank(%d) mod cudaDeviceCount(%d)) \n",
	     myrank, assignedCudaDevice, myrank, cudaDeviceCount );
      exit(-1);
    }

  printf("MPI Rank %d: leaving CPU land and going to CUDA Device %d \n", myrank, (myrank % cudaDeviceCount));

  Hello_kernel<<<1,1>>>( myrank );

  cudaDeviceSynchronize();

  printf("MPI Rank %d: re-entering CPU land \n", myrank );
}

__global__ void Hello_kernel( int myrank )
{
  cudaError_t cE = cudaSuccess;
  int device=-12;
  int cudaDeviceCount = -1;

  /* if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess ) */
  /*   { */
  /*     printf(" Rank %d in CUDA: Unable to determine cuda device count, error is %d, count is %d\n",  */
  /* 	     myrank, cE, cudaDeviceCount ); */
  /*   } */

  if( (cE = cudaGetDevice( &device)) != cudaSuccess )
    {
      printf(" Rank %d in CUDA: Unable to determine cuda device number, error is %d, device is %d\n", 
	     myrank, cE, device );
    }

  printf("Hello World from CUDA/MPI: Rank %d, Device %d, Thread %d, Block %d \n",
	 myrank, device, threadIdx.x, blockIdx.x );

  __syncthreads();
}
