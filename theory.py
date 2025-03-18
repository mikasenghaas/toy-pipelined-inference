def get_total_time(num_new_tokens, num_micro_batches, world_size, forward_time, latency):
    latency /= 1000
    forwarding_all_micro_batches = num_micro_batches * forward_time
    waiting_for_next_micro_batch = world_size * (forward_time + latency)
    decoding = num_new_tokens * max(forwarding_all_micro_batches, waiting_for_next_micro_batch)
    bubble = (world_size - 1) * (forward_time + latency) - abs(forwarding_all_micro_batches - waiting_for_next_micro_batch)
    return decoding + bubble

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-new-tokens", type=int)
    parser.add_argument("--num-micro-batches", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--forward-time", type=float)
    parser.add_argument("--latency", type=float)
    args = parser.parse_args()

    print(get_total_time(args.num_new_tokens, args.num_micro_batches, args.world_size, args.forward_time, args.latency))