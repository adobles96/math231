import numpy as np
from scipy.special import comb

p=4/9

def partial_E(n):
    s = 0
    for i in range(n):
        s += comb(3*i, i)*(p**i)*(1-p)**(2*n)
    return s

def calc_E(m, j0):
    N = 0
    for i in range(m):
        y = 0
        n = 1
        while True:
            if np.random.uniform() < p:
                y -= 2
            else:
                y += 1
            if y <= -j0:
                break
            n += 1
        N += n
    return N/m

if __name__ == '__main__':
    print(calc_E(10000, 1))
