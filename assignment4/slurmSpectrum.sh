module load xl_r spectrum-mpi

srun hostname -s | sort -u > /tmp/hosts.$SLURM_JOB_ID

taskset --cpu-list 0,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124 mpirun -hostfile /tmp/hosts.$SLURM_JOB_ID -np 32 ./assignment4-exe scratch/test.txt

rm /tmp/hosts.$SLURM_JOB_ID

salloc --gres:gpu:1,nvme -t 30