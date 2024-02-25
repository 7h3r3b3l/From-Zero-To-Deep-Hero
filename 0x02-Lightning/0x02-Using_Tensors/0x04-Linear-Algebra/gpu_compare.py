import torch
import time
import numpy as np
from tqdm import tqdm


def gpu_mul():
    t1 = torch.rand(10_000,10_000).to("cuda")
    t2 = torch.rand(10_000,10_000).to("cuda")
    start = time.time()
    final = torch.matmul(t1,t2)
    end = time.time()
    total = end - start
    return total


def cpu_mul():
    t1 = torch.rand(10_000,10_000)
    t2 = torch.rand(10_000,10_000)
    start = time.time()
    final = torch.matmul(t1,t2)
    end = time.time()
    total = end - start
    return total



gpus = [gpu_mul() for i in tqdm(range(100))]
cpus = [cpu_mul() for i in tqdm(range(100))]

cpu = np.mean(cpus)
gpu = np.mean(gpus)
print(f"cpu : {cpu}\ngpu: {gpu}")
print(f"gpu is {cpu/gpu} times faster than cpu")
