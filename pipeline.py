import time
import sys
import torch
import torch.distributed as dist
import multiprocessing as mp
import threading
from functools import partial
from typing import Optional
from lovely_tensors import monkey_patch; monkey_patch()
from loguru import logger as loguru_logger

next_token = 0
logger = None
def setup_logger(rank: int, log_level: str, start_time: Optional[float] = None):
    loguru_logger.remove()
    colors = ["green", "blue", "yellow", "red"]
    color = colors[rank]
    
    def my_format(record):
        if start_time is not None:
            # Calculate elapsed time from custom start time
            current_time = record["time"].timestamp()
            elapsed = current_time - start_time
        else:
            # Use loguru's default elapsed time
            elapsed = record["elapsed"].total_seconds()
            
        return f"<level><{color}>Rank {rank} | {{level:<5}} | {{time:mm:ss}} | {elapsed:.1f}s | {{message}}</{color}></level>\n"
    
    loguru_logger.add(
        sys.stdout,
        format=my_format,
        colorize=True,
        enqueue=True,
        level=log_level,
    )
    global logger
    logger = loguru_logger.bind(rank=rank)

def init_model(rank, world_size, forward_time: float = 1.0):
    def run_forward(x, token_idx, micro_batch_idx, rank, world_size):
        global next_token
        if logger is not None:
            logger.debug(f"Run forward ({token_idx}, {micro_batch_idx})")
        time.sleep(forward_time)
        if rank != world_size - 1:
            output = torch.full((x.size(0), 1), x.float().mean(), dtype=torch.long)
        else:
            next_token += 1
            output = torch.full((x.size(0), 1), next_token, dtype=torch.long)
        if logger is not None:
            logger.info(f"Ran forward ({token_idx}, {micro_batch_idx})")
        return output
    return partial(run_forward, rank=rank, world_size=world_size)

def warmup(model, batched_tokens, rank, world_size):
    tokens_shape = (batched_tokens[0].size(0), 1)
    hidden_states_shape = (batched_tokens[0].size(0), 1, 4096)
    dtype = batched_tokens[0].dtype # torch.long
    device = batched_tokens[0].device # cpu
    if rank == 0: # first stage
        hidden_states = model(batched_tokens[0], 0, 0)
        dist.send(hidden_states, dst=rank+1, tag=0)
        next_tokens = torch.empty(tokens_shape, dtype=dtype, device=device)
        dist.recv(next_tokens, src=world_size - 1, tag=0)
    elif rank == world_size - 1: # last stage
        hidden_states = torch.empty(hidden_states_shape, dtype=dtype, device=device)
        dist.recv(hidden_states, src=rank - 1, tag=0)
        next_tokens = model(hidden_states, 0, 0)
        dist.send(next_tokens, dst=0, tag=0)
    else:
        hidden_states = torch.empty(hidden_states_shape, dtype=dtype, device=device)
        dist.recv(hidden_states, src=rank - 1, tag=0)
        hidden_states = model(hidden_states, 0, 0)
        dist.send(hidden_states, dst=rank + 1, tag=0)

def pipeline1(model, batched_tokens, rank, world_size, num_new_tokens, latency: float = 0.0):
    """Synchronous 1F1B pipeline"""
    tokens_shape = (batched_tokens[0].size(0), 1)
    hidden_states_shape = (batched_tokens[0].size(0), 1, 1)
    dtype = batched_tokens[0].dtype # torch.long
    device = batched_tokens[0].device # cpu

    if rank == 0: # first stage
        for i in range(num_new_tokens): # decoding steps
            for j in range(len(batched_tokens)):
                hidden_states = model(batched_tokens[j], i, j)
                time.sleep(latency / 1000)
                dist.send(hidden_states, dst=1, tag=j)
                next_tokens = torch.empty(tokens_shape, dtype=dtype, device=device)
                dist.recv(next_tokens, src=world_size - 1, tag=j)
    elif rank == world_size - 1: # last stage
        for i in range(num_new_tokens): # decoding steps
            for j in range(len(batched_tokens)):
                hidden_states = torch.empty(hidden_states_shape, dtype=dtype, device=device)
                dist.recv(hidden_states, src=rank-1, tag=j)
                next_tokens = model(hidden_states, i, j)
                time.sleep(latency / 1000)
                dist.send(next_tokens, dst=0, tag=j)
    else: # intermediate stages
        for i in range(num_new_tokens): # decoding steps
            for j in range(len(batched_tokens)):
                hidden_states = torch.empty(hidden_states_shape, dtype=dtype, device=device)
                dist.recv(hidden_states, src=rank - 1, tag=j)
                hidden_states = model(hidden_states, i, j)
                time.sleep(latency / 1000)
                dist.send(hidden_states, dst=rank + 1, tag=j)

def pipeline2(model, batched_tokens, rank, world_size, num_new_tokens, latency: float = 0.0):
    """Synchronous AFAB pipeline"""
    tokens_shape = (batched_tokens[0].size(0), 1)
    hidden_states_shape = (batched_tokens[0].size(0), 1, 1)
    dtype = batched_tokens[0].dtype # torch.long
    device = batched_tokens[0].device # cpu
    
    if rank == 0: # first stage
        for i in range(num_new_tokens): # decoding steps
            for j in range(len(batched_tokens)):
                hidden_states = model(batched_tokens[j], i, j)
                time.sleep(latency / 1000)
                dist.send(hidden_states, dst=1, tag=j)
            
            # Now receive all results
            for j in range(len(batched_tokens)):
                next_tokens = torch.empty(tokens_shape, dtype=dtype, device=device)
                dist.recv(next_tokens, src=1, tag=j)
    
    elif rank == world_size - 1: # last stage
        for i in range(num_new_tokens): # decoding steps
            all_hidden_states = []
            for j in range(len(batched_tokens)):
                hidden_states = torch.empty(hidden_states_shape, dtype=dtype, device=device)
                dist.recv(hidden_states, src=0, tag=j)
                all_hidden_states.append(hidden_states)
            for j in range(len(batched_tokens)):
                next_tokens = model(all_hidden_states[j], i, j)
                time.sleep(latency / 1000)
                dist.send(next_tokens, dst=0, tag=j)
    else:
        for i in range(num_new_tokens): # decoding steps
            all_hidden_states = []
            for j in range(len(batched_tokens)):
                hidden_states = torch.empty(hidden_states_shape, dtype=dtype, device=device)
                dist.recv(hidden_states, src=0, tag=j)
                all_hidden_states.append(hidden_states)
            for j in range(len(batched_tokens)):
                hidden_states = model(all_hidden_states[j], i, j)
                time.sleep(latency / 1000)
                dist.send(hidden_states, dst=1, tag=j)

def delayed_send(tensor, dst, tag, delay_ms):
    """Helper function to send tensor after specified delay using asyncio"""
    time.sleep(delay_ms / 1000)
    return dist.isend(tensor, dst=dst, tag=tag)

def pipeline3(model, batched_tokens, rank, world_size, num_new_tokens, latency: float = 0.0):
    """Asynchronous pipeline"""
    tokens_shape = (batched_tokens[0].size(0), 1)
    hidden_states_shape = (batched_tokens[0].size(0), 1, 1)
    dtype = batched_tokens[0].dtype # torch.long
    device = batched_tokens[0].device # cpu
    num_micro_batches = len(batched_tokens)

    dist.barrier()
    if rank == 0: # first stage
        # Initial forwards for all micro-batches to start the pipeline
        recv_reqs = [None] * num_micro_batches
        recv_buffers = [None] * num_micro_batches
        send_threads = []
        if num_new_tokens > 0:
            for j in range(num_micro_batches):
                hidden_states = model(batched_tokens[j], 0, j)
                thread = threading.Thread(
                    target=delayed_send,
                    args=(hidden_states.clone(), 1, j, latency)  # Clone tensor to ensure it persists
                )
                thread.start()
                send_threads.append(thread)
                # dist.isend(hidden_states, dst=1, tag=j)
                recv_buffers[j] = torch.empty(tokens_shape, dtype=dtype, device=device)
                # logger.debug(f"Scheduling recv for micro-batch {j}")
                recv_reqs[j] = dist.irecv(recv_buffers[j], src=world_size - 1, tag=j)

        # Process remaining tokens while keeping pipeline full
        for i in range(num_new_tokens): # decoding steps
            # Wait for oldest results and update
            for j in range(num_micro_batches):
                # logger.debug(f"Waiting for micro-batch {j}")
                recv_reqs[j].wait()
                # logger.debug(f"Received micro-batch {j}")
                batched_tokens[j] = torch.cat((batched_tokens[j], recv_buffers[j]), dim=1)
                
                # Immediately process and send next token, except on last iteration
                if i < num_new_tokens - 1:
                    hidden_states = model(batched_tokens[j], i, j)
                    thread = threading.Thread(
                        target=delayed_send,
                        args=(hidden_states.clone(), 1, j, latency)  # Clone tensor to ensure it persists
                    )
                    thread.start()
                    send_threads.append(thread)
                    # dist.isend(hidden_states, dst=1, tag=j)
                    recv_buffers[j] = torch.empty(tokens_shape, dtype=dtype, device=device)
                    # logger.debug(f"Scheduling recv for micro-batch {j}")
                    recv_reqs[j] = dist.irecv(recv_buffers[j], src=world_size - 1, tag=j)

        for thread in send_threads:
            thread.join()
    
    elif rank == world_size - 1: # last stage
        recv_reqs = [None] * num_micro_batches
        recv_buffers = [None] * num_micro_batches
        send_threads = []
        for j in range(num_micro_batches):
            recv_buffers[j] = torch.empty(hidden_states_shape, dtype=dtype, device=device)
            recv_reqs[j] = dist.irecv(recv_buffers[j], src=rank-1, tag=j)

        # Process tokens while keeping pipeline full
        for i in range(num_new_tokens): # decoding steps
            for j in range(num_micro_batches):
                # Wait for input
                # logger.debug(f"Waiting for micro-batch {j}")
                recv_reqs[j].wait()
                # logger.debug(f"Received micro-batch {j}")
                next_tokens = model(recv_buffers[j], i, j)
                thread = threading.Thread(
                    target=delayed_send,
                    args=(next_tokens.clone(), 0, j, latency)
                )
                thread.start()
                send_threads.append(thread)
                recv_buffers[j] = torch.empty(hidden_states_shape, dtype=dtype, device=device)
                recv_reqs[j] = dist.irecv(recv_buffers[j], src=rank-1, tag=j)
        for thread in send_threads:
            thread.join()
    else:
        recv_reqs = [None] * num_micro_batches
        recv_buffers = [None] * num_micro_batches
        send_threads = []
        for j in range(num_micro_batches):
            recv_buffers[j] = torch.empty(hidden_states_shape, dtype=dtype, device=device)
            recv_reqs[j] = dist.irecv(recv_buffers[j], src=rank-1, tag=j)

        for i in range(num_new_tokens): # decoding steps
            for j in range(num_micro_batches):
                recv_reqs[j].wait()
                hidden_states = model(recv_buffers[j], i, j)
                thread = threading.Thread(
                    target=delayed_send,
                    args=(hidden_states.clone(), rank+1, j, latency)
                )
                thread.start()
                send_threads.append(thread)
                recv_buffers[j] = torch.empty(hidden_states_shape, dtype=dtype, device=device)
                recv_reqs[j] = dist.irecv(recv_buffers[j], src=rank-1, tag=j)
        for thread in send_threads:
            thread.join()
    
    dist.barrier()
            

PIPELINES = {
    "1f1b": pipeline1,
    "afab": pipeline2,
    "async": pipeline3,
}

def get_pipeline(pipeline_name: str):
    assert pipeline_name in PIPELINES, f"Invalid pipeline name: {pipeline_name}"
    return PIPELINES[pipeline_name]

def main(rank, args):
    # Initialize process group
    dist.init_process_group(backend="gloo", init_method="tcp://localhost:12345", rank=rank, world_size=args.world_size)

    # Prepare dummy tokens
    batch_size = args.world_size * args.num_micro_batches
    tokens = torch.zeros(batch_size, args.prompt_tokens, dtype=torch.long)

    # Split tokens into micro-batches
    micro_batch_size = batch_size // args.num_micro_batches
    micro_batches = list(tokens.split(micro_batch_size, dim=0))

    # Initialize model
    model = init_model(rank, args.world_size, args.forward_time)

    # Warmup
    warmup(model, micro_batches, rank, args.world_size)

    # Run pipelined decoding
    dist.barrier()
    setup_logger(rank, args.log_level, time.time())
    logger.info("Running")
    get_pipeline(args.pipeline)(model, micro_batches, rank, args.world_size, args.num_new_tokens, args.latency)
    logger.info("Done")

    dist.destroy_process_group()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=str, default="1f1b", help="Pipeline function to use")
    parser.add_argument("--prompt-tokens", type=int, default=1, help="Number of prompt tokens")
    parser.add_argument("--num-new-tokens", type=int, default=3, help="Number of new tokens to generate")
    parser.add_argument("--num-micro-batches", type=int, default=2, help="Number of micro-batches")
    parser.add_argument("--world-size", type=int, default=2, help="Number of processes to run")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    parser.add_argument("--forward-time", type=float, default=1.0, help="Inference time of the model (s)")
    parser.add_argument("--latency", type=float, default=0, help="Latency of the network (ms)")
    args = parser.parse_args()

    if args.num_micro_batches is None:
        args.num_micro_batches = args.world_size

    # Start processes
    processes = [mp.Process(target=main, args=(rank, args)) for rank in range(args.world_size)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()