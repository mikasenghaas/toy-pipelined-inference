import time
from time import perf_counter
import torch.distributed as dist
import multiprocessing as mp
import torch

def main(rank: int, world_size: int):
    dist.init_process_group(backend="gloo", init_method="tcp://localhost:12345", rank=rank, world_size=world_size)
    dist.barrier()
    send_times = []
    recv_times = []
    if rank == 0:
        for i in range(10):
            x = torch.tensor([1, 2, 3], dtype=torch.float32)
            start = perf_counter()
            dist.send(x, dst=1)
            end = perf_counter()
            send_times.append((end - start) * 1000)
        for i in range(10):
            x = torch.tensor([1, 2, 3], dtype=torch.float32)
            start = perf_counter()
            dist.recv(x, src=world_size - 1)
            end = perf_counter()
            recv_times.append((end - start) * 1000)
    elif rank == world_size - 1:
        for i in range(10):
            x = torch.empty(3, dtype=torch.float32)
            start = perf_counter()
            dist.recv(x, src=rank - 1)
            end = perf_counter()
            recv_times.append((end - start) * 1000)
        for i in range(10):
            start = perf_counter()
            dist.send(x, dst=0)
            end = perf_counter()
            send_times.append((end - start) * 1000)
    else:
        for _ in range(10):
            x = torch.empty(3, dtype=torch.float32)
            start = perf_counter()
            dist.recv(x, src=rank - 1)
            end = perf_counter()
            recv_times.append((end - start) * 1000)
        for _ in range(10):
            start = perf_counter()
            dist.send(x, dst=rank + 1)
            end = perf_counter()
            send_times.append((end - start) * 1000)
    recv_times.pop(0)
    send_times.pop(0)
    print(f"[Rank {rank}] Send took {sum(send_times) / len(send_times):.2f} ms")
    print(f"[Rank {rank}] Recv took {sum(recv_times) / len(recv_times):.2f} ms")

if __name__ == "__main__":
    world_size = 3
    processes = [mp.Process(target=main, args=(rank, world_size)) for rank in range(world_size)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
