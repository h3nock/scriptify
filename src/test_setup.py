import torch
print("torch.version.cuda    =", torch.version.cuda) # type: ignore
print("torch.backends.cudnn.version() =", torch.backends.cudnn.version())
print("torch.cuda.device_count() =", torch.cuda.device_count())

# test_spawn.py
import torch.multiprocessing as mp

def simple_function(rank):
    print(f"Hello from rank {rank}")

if __name__ == "__main__":
    mp.spawn(simple_function, args=(), nprocs=2, join=True) # type: ignore