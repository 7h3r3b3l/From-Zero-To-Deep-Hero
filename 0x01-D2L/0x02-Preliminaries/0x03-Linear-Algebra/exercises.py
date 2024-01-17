# Exercises from section 2.3 - Matices in d2l

import random
import torch
from tqdm import tqdm


## 1. Prove that (A.T).T == A

# lol this is not a prove but lol
def generate_matrix():
    i = random.randint(20,1000)
    j = random.randint(20,1000)
    m = torch.rand([i,j])
    return m


# Prove by bruteforce hahaha
def prove():
    print('proving... (lol)')
    for i in tqdm(range(10_000)):
        m = generate_matrix()
        m_transpose = m.T
        m_transpose_transpose = m_transpose.T
        if(not torch.equal(m_transpose_transpose , m)):
            return False
    print("Proved! xd")
    return True
    
prove()
