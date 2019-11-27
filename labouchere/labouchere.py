import numpy.random as random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
import matplotlib

p = 0.4
q = 1 - p
a_config = 'profit' # 'profit': A(l) = b(l)(2p-1) -- 'total bet': A(l) = b(l) -- 'coups': A(l) = 1
M = 5


def b(l):
    """
    Returns the size of the next Labouchere bet given a current list l.

    :param l: The current list

    :return: The size of the next wager.
    """
    if not l:
        return 0
    elif len(l) == 1:
        return l[0]
    else:
        return l[0] + l[-1]


def b_fib(l):
    """
    Returns the size of the next Fibonacci bet given a current list l.

    :param l: The current list

    :return: The size of the next wager.
    """
    if not l:
        return 0
    elif len(l) == 1:
        return l[0]
    else:
        return l[-2] + l[-1]


def new_l(l, X, inplace=False):
    """
    Updates the list l according to the Labouchere betting system.

    :param l: current list
    :param X: outcome of last coup
    :param inplace: If true list is modified in-place, otherwise a new list is created and returned

    :return:  updated list or None if inplace=True
    """
    if not l:
        return l
    if inplace:
        if X == 1:
            l.pop()
            if l:
                del(l[0])
        else:
            l.append(b(l))
        return
    if X == 1:
        return l[1:-1]
    else:
        return l + [b(l)]


def new_l_fib(l, X, inplace=False):
    """
    Updates the list l according to the Fibonacci betting system.

    :param l: current list
    :param X: outcome of last coup
    :param inplace: If true list is modified in-place, otherwise a new list is created and returned

    :return:  updated list or None if inplace=True
    """
    if not l:
        return l
    if inplace:
        if X == 1:
            l.pop()
            if l:
                del(l[0])
        else:
            l.append(b_fib(l))
        return
    if X == 1:
        return l[:-2]
    else:
        return l + [b_fib(l)]


def A(l):
    """
    A function that depends on the constant a_config. To be used in computing R(l).
    When a_config = 'profit', R(l) will give the expected cumulative profit if gambler starts with list l.
    When a_config = 'total bet', R(l) will give the expected total amount bet if gambler starts with list l.
    When a_config = 'coups', R(l) will give the expected number of coups if gambler starts with list l.

    :param l: the current list

    :return: see above.
    """
    if a_config == 'profit':
        return b(l)*(2*p - 1)
    elif a_config == 'total bet':
        return b(l)
    return 1


def Q(l):
    """
    Wrapper function for recursive function _Q.
    Computes the probability that a Labouchere list becomes empty before a betting limit M is reached.
    :param l: starting list


    :return: the probability the the list becomes empty before a betting limit M is reached.
    """
    return _Q(l, set(), {})[0]

def _Q(l, seen, d):
    """
    Utility recursive function for computing the probability that a Labouchere list becomes empty before a betting limit
    M is reached.

    :param l: Current list.
    :param seen: A set keeping track of previously seen values of l.
    :param d: A memoization dictionary to keep track of previously computed values of Q(l).

    :return: A tuple of
    """
    if not l:
        return 1, {}
    if b(l) > M:
        return 0, {}
    key = str(l)
    if key in d:
        return d[key], {}
    if key in seen:
        return 0, {key : 1}
    seen.add(key)
    left_total, left_unks = _Q(new_l(l, -1), seen, d)
    right_total, right_unks = _Q(new_l(l, 1), seen, d)
    total = q*left_total + p*right_total
    c = q*left_unks.pop(key, 0) + p*right_unks.pop(key, 0)
    unks = {}
    for u in left_unks.keys() | right_unks.keys():
        unks[u] = (q*left_unks.get(u, 0) + p*right_unks.get(u, 0)) / (1 - c)
    total /= (1 - c)
    if not unks:
        d[key] = total
    return total, unks





def R(l):
    """
    Wrapper function for recursive function _R.
    Computes relevant values related to the Labouchere betting system. See return for a more detailed explanation.

    :param l: Initial list.

    :return: When a_config = 'profit', R(l) will give the expected cumulative profit if gambler starts with list l.
    When a_config = 'coups', R(l) will give the expected number of coups if gambler starts with list l.
    When a_config = 'total bet', R(l) will give the expected total amount bet if gambler starts with list l.
    """
    return _R(l, set(), {})[0]


def _R(l, seen, d):
    """
    Utitlity recursive function to compute relevant values related to the Labouchere system. See docstring for R().

    :param l: Current list
    :param seen: A set that keeps track of previously seen values of l
    :param d: A memoization dictionary to keep track of previously computed values of R(l)

    :return: A tuple of (total, unks). Total is the value of R(l) minus any unknowns seen during its computation.
    Unks is a dictionary mapping these unknown values of R(.) to their coefficients.
    """
    if not l or b(l) > M:
        return 0, {}
    key = str(l)
    if key in d:
        return d[key], {}
    if key in seen:
        return 0, {key : 1}
    seen.add(key)
    left_total, left_unks = _R(new_l(l, -1), seen, d)
    right_total, right_unks = _R(new_l(l, 1), seen, d)
    total = A(l) + q*left_total + p*right_total
    c = q*left_unks.pop(key, 0) + p*right_unks.pop(key, 0)
    unks = {}
    for u in left_unks.keys() | right_unks.keys():
        unks[u] = (q*left_unks.get(u, 0) + p*right_unks.get(u, 0)) / (1 - c)
    total /= (1 - c)
    if not unks:
        d[key] = total
    return total, unks


def simulate(m, l, f0):
    """
    Simulates a run of the Labouchere system at an even-money game, starting with list l and initial fortune f0.
    The simulation stops after n coups, or after the list is empty, or after the maximum bet M is exceeded.

    :param m: Specifies the maximum number of coups to play. If set to -1 then the simulation will be carried out
    indefinitely until either l becomes empty or the maximum bet M is exceeded
    :param l: specifies the initial list
    :param f0: specifies the initial fortune

    :return: A tuple of ({B_n}, {L_n}, {F_n}, l, n) where the first is a list of bets made, the second is a list with
    the length of the Labouchere list after each coup, the third is a list of the fortune of the gambler after each coup,
    the fourth is the final list, ie the one at termination, and n is the number of coups played.
    """
    B = []
    L = []
    F = [f0]
    n = 0
    while n < m or m < 0:
        bet = b(l)
        if bet > M:
            print('Maximum bet size exceeded.')
            return (B, L, F, l, n)
        if bet == 0:
            print('List emptied.')
            return (B, L, F, l, n)
        B.append(bet)
        if random.rand() < p:
            X = 1
        else: X = -1
        new_l(l, X, True)
        L.append(len(l))
        F.append(F[-1] + bet*X)
        n += 1
    print('Simulation completed.')
    return (B, L, F, l, n)

def gen_l(n, l0, absorbption):
    """
    Simulates an asymmetric random walk starting at l0 moving up 1 with prob q, and down 2 with prob p.

    :param n: number of steps in the stochastic process to simulate
    :param l0: initial value
    :param absorbption: stops generating if absorbing state hit
    :return:
    """
    l = [l0]
    l_i = l0
    for _ in range(n):
        if l_i <= absorbption:
            break
        if np.random.rand() < p:
            l_i = l_i - 2
            l.append(l_i)
        else:
            l_i = l_i + 1
            l.append(l_i)
    return l

def plot_sp(l):
    """
    Plots stochastic processes in time.

    :param l: A list of lists containing the values of a stochastic process
    :return: None
    """
    matplotlib.style.use('ggplot')
    fig, ax = plt.subplots()
    #cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(len(l)):
        ax.plot(l[i], color='k')#cycle[i % len(cycle)])
    _, x2, y1, y2 = plt.axis()
    plt.axis((0, x2, y1, y2))
    mean = np.mean([len(x) for x in l])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    txt = '$\mu=%.2f$' % (mean)
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    plt.show()

def make_hist(l):
    matplotlib.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.hist(l, bins=30, alpha=0.75, ec='black', histtype='bar')
    mean = np.mean(l)
    median = np.median(l)
    std = np.std(l)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    txt = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$' % (mean, median, std)
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    plt.show()


def simulate_max_bet(l, n):
    """
    Simulates n Labouchere games played with no betting limit, infinite credit, and initial list l.

    :param l: The initial list.

    :return: An n-element list where the ith element is the maximum bet size of the ith game simulated.
    """
    B_stars = []
    for i in range(1, n + 1):
        if i % 100000 == 0:
            print('{}th simulation completed.'.format(i))
        B_star = 1
        curr_l = l.copy()
        while curr_l:
            bet = b(curr_l)
            if bet > B_star:
                B_star = bet
            if random.rand() < p:
                X = 1
            else: X = -1
            new_l(curr_l, X, True)
        B_stars.append(B_star)
    print('Simulation completed.')
    return B_stars


def simulate_stopped_max_bets(l, n, st):
    """
    Simulates n runs of the stochastic process {B*_i} each element of which is generated as in simulate_max_bet, but
    this time without stopping until the criteria for a stopping time st is met.

    :param l: The initial list.
    :param n: The number of times the stochastic process {B*_i} will be generated till stopping.
    :param st: A function with boolean return value. Signature bool st(list B_stars).

    :return: -stopped_b_stars: a list of the B* at the stopped time.
    -stopping_times: the stopping time at which the ith run was stopped.
    """
    stopped_b_stars = []
    stopping_times = []
    for i in range(1, n+1):
        B_stars = []
        stopped = False
        if i % 1000 == 0:
            print('{}th simulation completed.'.format(i))
        while not stopped:
            B_star = 1
            curr_l = l.copy()
            while curr_l:
                bet = b(curr_l)
                if bet > B_star:
                    B_star = bet
                if random.rand() < p:
                    X = 1
                else:
                    X = -1
                new_l(curr_l, X, True)
            B_stars.append(B_star)
            stopped = st(B_stars)
        stopped_b_stars.append(B_star)
        stopping_times.append(len(B_stars))
    print('Simulation completed.')
    return stopped_b_stars, stopping_times


def run_simul_1():
    B, L, F, l, n = simulate(500, [1, 2, 3, 4], 50)
    print('Coups played = {}'.format(n))
    print('Final profit = {}'.format(F[-1] - F[0]))
    print('Final list: {}'.format(l))
    make_hist(B)
    make_hist(L)
    make_hist(F)


def run_simul_2(num_sims):
    B_stars = simulate_max_bet([1], num_sims)
    print('Maximum B* = {}'.format(max(B_stars)))
    print('Mean B* = {}'.format(np.mean(B_stars)))
    print('Median B* = {}'.format(np.median(B_stars)))
    print('Std dev B* = {}'.format(np.std(B_stars)))


C = 10000

def st(l):
    return l[-1] == 1 or l[-1] >= C


def run_simul_3(num_sims):
    stopped_b_stars, stopping_times = simulate_stopped_max_bets([1], num_sims, st)
    print('Maximum Stopped B* = {}'.format(max(stopped_b_stars)))
    print('Mean Stopped B* = {}'.format(np.mean(stopped_b_stars)))
    print('Proportion of B*=1: {}'.format(stopped_b_stars.count(1)/len(stopped_b_stars)))
    make_hist(stopped_b_stars)
    probs = [x/stopping_times.count(x) for x in set(stopping_times)]
    geom = [x/stats.geom.pmf(x, p) for x in range(min(stopping_times), max(stopping_times) + 1)]
    kl = stats.entropy(probs, geom)
    print('KL divergence between stopping times and geometric dist. = {}'.format(kl))
    make_hist(stopping_times)

def run_simul_4(num_sims, length, l0):
    l = [gen_l(length, l0, 0) for _ in range(num_sims)]
    plot_sp(l)
    make_hist([len(x) for x in l])


def test_Q_and_R():
    start_time = time.time()
    print(Q([1]))
    end_time = time.time()
    print("--- Completed in {:0.6f} seconds ---".format(end_time - start_time))
    start_time = time.time()
    print(R([1]))
    end_time = time.time()
    print("--- Completed in {:0.3f} seconds ---".format(end_time - start_time))


if __name__ == '__main__':
   run_simul_2(1000000)
