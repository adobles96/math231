import numpy as np
import matplotlib.pyplot as plt

def next_kolakoski(k, block):
    # k is a list of partial
    if block == 0 or not k:
        k += [1]
    if block % 2 == 0:
        # Indicates block is even so next block is a 2-block
        k += [2]*k[block]
    else:
        # Indicates block is odd so next block is a 1-block
        k += [1]*k[block]

def calc_Zn(k, a, b):
    # Need to address numerical instability
    a = np.float64(a)
    b = np.float64(b)
    A = ((a/b)**k.count(1))
    B = ((1-a)/(1-b))**k.count(2)
    # The computation is done weirdly to provide some numerical stability
    # Maybe to improve it we could track log(Zn) in which case its limit should be -infty
    return A*B

def gen_kolakoski(n):
    k = []
    for i in range(n):
        next_kolakoski(k, i)
    return k

def gen_Zn(n, k, a, b):
    z = []
    for i in range(1, n+1):
        z.append(calc_Zn(k[:i], a, b))
    return z

def plot_Zn(z):
    plt.figure()
    plt.plot(z)
    plt.show()

if __name__ == '__main__':
    k = gen_kolakoski(5000)
    z = gen_Zn(5000, k, 0.49, 0.501)
    print('Minimum Zn = {}'.format(min(z)))
    print('Last Zn = {}'.format(z[-1]))
    plot_Zn(z[::5])
    plot_Zn(z[-500:])
    # k = gen_kolakoski(10000)
    # print(k.count(1)/len(k))
