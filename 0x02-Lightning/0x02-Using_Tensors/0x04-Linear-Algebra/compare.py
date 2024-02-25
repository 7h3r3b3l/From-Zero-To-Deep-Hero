import torch
import matplotlib.pyplot as plt
import random
import time
import termplotlib as tpl
from tqdm import tqdm



def plain_multiplication(size):
    x = [random.randint(0,10) for i in range(size)]
    w = [random.uniform(0,1) for i in range(size)]
    start = time.time()
    b = 0
    output = 0
    for x_i,w_i in zip(x,w):
        output += (x_i * w_i)
    end = time.time()
    return end-start


def torch_multiplication(size):
    x = torch.rand(size)
    w = torch.rand(size)
    b = torch.tensor(1)
    start = time.time()
    # This can be accelerated even more on GPU
    output = x.dot(w)
    end = time.time()
    return end-start


def torch_multiplication_gpu(size):
    x = torch.rand(size).to("cuda")
    w = torch.rand(size).to("cuda")
    b = torch.tensor(1)
    start = time.time()
    # This can be accelerated even more on GPU
    output = x.dot(w)
    end = time.time()
    return end-start

test_range = range(1,100_000)
torch_times_gpu = [torch_multiplication_gpu(i) for i in tqdm(test_range)]
torch_times = [torch_multiplication(i) for i in tqdm(test_range)]
plain_times = [plain_multiplication(i) for i in tqdm(test_range)]

fig = tpl.figure()
fig.plot(test_range, plain_times)
fig.plot(test_range, torch_times)
fig.plot(test_range, torch_times_gpu)
fig.show()
