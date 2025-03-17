import time
import sys
import torch
import torch.distributed as dist
import multiprocessing as mp
from functools import partial
from subprocess import Popen
from lovely_tensors import monkey_patch; monkey_patch()
from loguru import logger as loguru_logger

next_token = 0
logger = None
def setup_logger(rank: int):
    loguru_logger.remove()
    colors = ["green", "blue", "yellow", "red"]
    color = colors[rank]
    start_time = time.time()
    loguru_logger.add(
        sys.stdout,
        format=f"<level><{color}>Rank {rank} | {{level:<5}} | {{time:mm:ss}} | {{elapsed.seconds:>3.1f}}s | {{message}}</{color}></level>",
        colorize=True,
        enqueue=True,
        level="INFO",
    )
    global logger
    logger = loguru_logger.bind(rank=rank)

def init_model(rank, world_size):
    def run_forward(x, token_idx, micro_batch_idx, rank, world_size):
        global next_token
        logger.info(f"Running forward for token {token_idx} of micro batch {micro_batch_idx}")
        time.sleep(1)
        if rank != world_size - 1:
            output = torch.full((x.size(0), 1, 4096), x.float().mean(), dtype=torch.long)
        else:
            next_token += 1
            output = torch.full((x.size(0), 1), next_token, dtype=torch.long)
        logger.info(f"Ran forward for token {token_idx} of micro batch {micro_batch_idx}")
        return output
    return partial(run_forward, rank=rank, world_size=world_size)

def pipeline1(model, batched_tokens, rank, world_size, num_new_tokens):
    """Decodes micro-batches one-by-one ≈ O(num_micro_batches * num_new_tokens * num_devices)"""
    tokens_shape = (batched_tokens[0].size(0), 1)
    hidden_states_shape = (batched_tokens[0].size(0), 1, 4096)
    dtype = batched_tokens[0].dtype # torch.long
    device = batched_tokens[0].device # cpu

    if rank == 0: # first stage
        for i in range(num_new_tokens): # decoding steps
            for j in range(len(batched_tokens)):
                hidden_states = model(batched_tokens[j], i, j)
                dist.send(hidden_states, dst=1, tag=j)
                next_tokens = torch.empty(tokens_shape, dtype=dtype, device=device)
                dist.recv(next_tokens, src=1, tag=j)
                batched_tokens[j] = torch.cat((batched_tokens[j], next_tokens), dim=1)
    elif rank == world_size - 1: # last stage
        for i in range(num_new_tokens): # decoding steps
            for j in range(len(batched_tokens)):
                hidden_states = torch.empty(hidden_states_shape, dtype=dtype, device=device)
                dist.recv(hidden_states, src=0, tag=j)
                next_tokens = model(hidden_states, i, j)
                dist.send(next_tokens, dst=0, tag=j)
                batched_tokens[j] = torch.cat((batched_tokens[j], next_tokens), dim=1)
    else:
        raise NotImplementedError
    
    return batched_tokens


def pipeline2(model, batched_tokens, rank, world_size, num_new_tokens):
    """Forwards all micro-batches at once, but waits for all micro-batches to be ready"""
    tokens_shape = (batched_tokens[0].size(0), 1)
    hidden_states_shape = (batched_tokens[0].size(0), 1, 4096)
    dtype = batched_tokens[0].dtype # torch.long
    device = batched_tokens[0].device # cpu
    
    if rank == 0: # first stage
        for i in range(num_new_tokens): # decoding steps
            for j in range(len(batched_tokens)):
                hidden_states = model(batched_tokens[j], i, j)
                dist.send(hidden_states, dst=1, tag=j)
            
            # Now receive all results
            for j in range(len(batched_tokens)):
                next_tokens = torch.empty(tokens_shape, dtype=dtype, device=device)
                dist.recv(next_tokens, src=1, tag=j)
                batched_tokens[j] = torch.cat((batched_tokens[j], next_tokens), dim=1)
    
    elif rank == world_size - 1: # last stage
        for i in range(num_new_tokens): # decoding steps
            all_hidden_states = []
            for j in range(len(batched_tokens)):
                hidden_states = torch.empty(hidden_states_shape, dtype=dtype, device=device)
                dist.recv(hidden_states, src=0, tag=j)
                all_hidden_states.append(hidden_states)
            for j in range(len(batched_tokens)):
                next_tokens = model(all_hidden_states[j], i, j)
                dist.send(next_tokens, dst=0, tag=j)
                batched_tokens[j] = torch.cat((batched_tokens[j], next_tokens), dim=1)
    else:
        raise NotImplementedError
    
    return batched_tokens

def pipeline3(model, batched_tokens, rank, world_size, num_new_tokens):
    """Async recv on rank 0, else sync"""
    tokens_shape = (batched_tokens[0].size(0), 1)
    hidden_states_shape = (batched_tokens[0].size(0), 1, 4096)
    dtype = batched_tokens[0].dtype # torch.long
    device = batched_tokens[0].device # cpu
    num_micro_batches = len(batched_tokens)
    
    if rank == 0: # first stage
        # Initial forwards for all micro-batches to start the pipeline
        recv_reqs = [None] * num_micro_batches
        recv_buffers = [None] * num_micro_batches
        if num_new_tokens > 0:
            for j in range(num_micro_batches):
                hidden_states = model(batched_tokens[j], 0, j)
                dist.send(hidden_states, dst=1, tag=j)

                recv_buffers[j] = torch.empty(tokens_shape, dtype=dtype, device=device)
                recv_reqs[j] = dist.irecv(recv_buffers[j], src=1, tag=j)
        
        # Process remaining tokens while keeping pipeline full
        for i in range(num_new_tokens): # decoding steps
            # Wait for oldest results and update
            for j in range(num_micro_batches):
                recv_reqs[j].wait()
                batched_tokens[j] = torch.cat((batched_tokens[j], recv_buffers[j]), dim=1)
                
                # Immediately process and send next token, except on last iteration
                if i < num_new_tokens - 1:
                    hidden_states = model(batched_tokens[j], i, j)
                    dist.send(hidden_states, dst=1, tag=j)

                    recv_buffers[j] = torch.empty(tokens_shape, dtype=dtype, device=device)
                    recv_reqs[j] = dist.irecv(recv_buffers[j], src=1, tag=j)
    
    elif rank == world_size - 1: # last stage
        # Process tokens while keeping pipeline full
        for i in range(num_new_tokens): # decoding steps
            for j in range(num_micro_batches):
                # Wait for input
                hidden_states = torch.empty(hidden_states_shape, dtype=dtype, device=device)
                dist.recv(hidden_states, src=0, tag=j)
                next_tokens = model(hidden_states, i, j)
                dist.send(next_tokens, dst=0, tag=j)
                batched_tokens[j] = torch.cat((batched_tokens[j], next_tokens), dim=1)
    
    else:
        raise NotImplementedError
    
    return batched_tokens

def pipeline4(model, batched_tokens, rank, world_size, num_new_tokens):
    """Fully asynchronous. Only wait on items that are needed."""
    tokens_shape = (batched_tokens[0].size(0), 1)
    hidden_states_shape = (batched_tokens[0].size(0), 1, 4096)
    dtype = batched_tokens[0].dtype # torch.long
    device = batched_tokens[0].device # cpu
    num_micro_batches = len(batched_tokens)

    dist.barrier()
    if rank == 0: # first stage
        # Initial forwards for all micro-batches to start the pipeline
        recv_reqs = [None] * num_micro_batches
        recv_buffers = [None] * num_micro_batches
        if num_new_tokens > 0:
            for j in range(num_micro_batches):
                hidden_states = model(batched_tokens[j], 0, j)
                dist.isend(hidden_states, dst=1, tag=j)
                recv_buffers[j] = torch.empty(tokens_shape, dtype=dtype, device=device)
                logger.debug(f"{time.strftime('%X')}] Scheduling recv for micro-batch {j}")
                recv_reqs[j] = dist.irecv(recv_buffers[j], src=1, tag=j)

        # Process remaining tokens while keeping pipeline full
        for i in range(num_new_tokens): # decoding steps
            # Wait for oldest results and update
            for j in range(num_micro_batches):
                logger.debug(f"{time.strftime('%X')}] Waiting for micro-batch {j}")
                recv_reqs[j].wait()
                logger.debug(f"{time.strftime('%X')}] Received micro-batch {j}")
                batched_tokens[j] = torch.cat((batched_tokens[j], recv_buffers[j]), dim=1)
                
                # Immediately process and send next token, except on last iteration
                if i < num_new_tokens - 1:
                    hidden_states = model(batched_tokens[j], i, j)
                    dist.isend(hidden_states, dst=1, tag=j)
                    recv_buffers[j] = torch.empty(tokens_shape, dtype=dtype, device=device)
                    logger.debug(f"{time.strftime('%X')}] Scheduling recv for micro-batch {j}")
                    recv_reqs[j] = dist.irecv(recv_buffers[j], src=1, tag=j)
    
    elif rank == world_size - 1: # last stage
        recv_reqs = [None] * num_micro_batches
        recv_buffers = [None] * num_micro_batches
        for j in range(num_micro_batches):
            recv_buffers[j] = torch.empty(hidden_states_shape, dtype=dtype, device=device)
            logger.debug(f"{time.strftime('%X')}] Scheduling recv for micro-batch {j}")
            recv_reqs[j] = dist.irecv(recv_buffers[j], src=0, tag=j)

        # Process tokens while keeping pipeline full
        for i in range(num_new_tokens): # decoding steps
            for j in range(num_micro_batches):
                # Wait for input
                logger.debug(f"{time.strftime('%X')}] Waiting for micro-batch {j}")
                recv_reqs[j].wait()
                logger.debug(f"{time.strftime('%X')}] Received micro-batch {j}")
                next_tokens = model(recv_buffers[j], i, j)
                dist.isend(next_tokens, dst=0, tag=j)
                batched_tokens[j] = torch.cat((batched_tokens[j], next_tokens), dim=1)
                recv_buffers[j] = torch.empty(hidden_states_shape, dtype=dtype, device=device)
                recv_reqs[j] = dist.irecv(recv_buffers[j], src=0, tag=j)
    
    else:
        raise NotImplementedError
    
    return batched_tokens

def get_pipeline(pipeline_idx: int):
    if pipeline_idx == 1:
        return pipeline1
    elif pipeline_idx == 2:
        return pipeline2
    elif pipeline_idx == 3:
        return pipeline3
    elif pipeline_idx == 4:
        return pipeline4
    else:
        raise ValueError(f"Invalid pipeline index: {pipeline_idx}")

def main(rank: int, world_size: int, pipeline_name: str):
    setup_logger(rank)
    logger.info(f"Running")
    dist.init_process_group(backend="gloo", init_method="tcp://localhost:12345", rank=rank, world_size=world_size)

    # Choose pipeline
    pipeline = get_pipeline(pipeline_name)

    # Prepare dummy tokens
    batch_size = 2
    prompt_tokens = 1
    tokens = torch.zeros(batch_size, prompt_tokens, dtype=torch.long)

    # Split tokens into micro-batches
    num_micro_batches = world_size
    micro_batch_size = batch_size // num_micro_batches
    micro_batches = list(tokens.split(micro_batch_size, dim=0))

    # Initialize model
    model = init_model(rank, world_size)

    # Run pipelined decoding
    num_new_tokens = 1
    start_time = time.time()
    batched_tokens = pipeline(model, micro_batches, rank, world_size, num_new_tokens)
    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
    print(torch.cat(batched_tokens, dim=0))

    dist.destroy_process_group()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=int, default=1, help="Pipeline function to use")
    parser.add_argument("--latency", type=float, default=0, help="Add latency (ms)")
    parser.add_argument("--world-size", type=int, default=2, help="Number of processes to run")
    args = parser.parse_args()

    # Add latency using subprocess
    if args.latency > 0:
        Popen(["sudo", "tc", "qdisc", "add", "dev", "lo", "root", "netem", "delay", f"{args.latency}ms"])

    processes = [mp.Process(target=main, args=(rank, args.world_size, args.pipeline)) for rank in range(args.world_size)]

    for p in processes:
        p.start()

    # Remove latency using subprocess
    if args.latency > 0:
        Popen(["sudo", "tc", "qdisc", "del", "dev", "lo", "root", "netem"])

    for p in processes:
        p.join()