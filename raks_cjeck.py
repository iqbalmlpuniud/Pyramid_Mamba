import os
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='gloo', rank=rank, world_size=size)

    print(f"Rank {rank} started")

    dist.barrier()

    print(f"Rank {rank} reached the barrier")
    dist.destroy_process_group()

if __name__ == "__main__":
    size = 8  # Total number of processes
    mp.spawn(run, args=(size,), nprocs=size, join=True)