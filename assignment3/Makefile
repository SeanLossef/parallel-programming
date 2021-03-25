all: highlife-mpi.c highlife-cuda.cu
	mpixlc -O3 highlife-mpi.c -c -o highlife-mpi.o
	nvcc -O3 -arch=sm_70 highlife-cuda.cu -c -o highlife-cuda.o
	mpicc -O3 highlife-mpi.o highlife-cuda.o -o highlife-exe -L/usr/local/cuda-10.2/lib64/ -lcudadevrt -lcudart -lstdc++

# sbatch -N 2 --ntasks-per-node=6 --partition=dcs --gres=gpu:4 -t 30 ./mpirun-hello.sh