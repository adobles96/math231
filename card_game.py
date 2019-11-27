"""
I have a standard 52 card deck. One by one, I draw cards from the deck and place
them face up in front of you (without putting them back). At any point, you may
stop me and say a color. If the next two cards I draw both have that color,
you win. Otherwise, you lose. What is the optimal strategy?
"""

import numpy as np

def shuffle(deck):
    """
    Returns a shuffled deck
    """
    return np.random.permutation(deck)

def play(n, strategy):
    """
    Simulates n games played according to strategy.
    """
    deck = [0]*26 + [1]*26
    outcomes = np.zeros((n, 3)) # holds tuples of (stopped at k, guess, won/lost)
    for i in range(n):
        deck = shuffle(deck)
        R = 0
        for k in range(52, 1, -1):
            move = strategy(k, R)
            if move != -1:
                outcomes[i] = [k, move, int(deck[k] == deck[k - 1] == move)]
            R += deck[k - 1]
    return outcomes

def p(k, r_rem):
    """
    Returns the probability of winning if we stop with k cards remaining.

    :param k: Cards remaining on deck
    :param r_rem: number of red cards remaining
    """
    r = max(r_rem, k - r_rem)
    return (r * (r - 1))/(k * (k - 1))

def s1(k, R):
    """
    Returns 1 if we guess two reds are comming out, 0 if two blacks, and -1 if
    we want to see another card.

    :param k: Cards remaining on deck
    :param R: number of red cards seen so far
    """
    if k==2:
        return int(R < 25)
    r_rem = 26 - R
    Ep_next = (r_rem/k)*p(k - 1, r_rem - 1) + ((k-r_rem)/k)*p(k - 1, r_rem)
    print(p(k, r_rem), Ep_next)
    if p(k, r_rem) > Ep_next:
        return int(r_rem > k - r_rem)
    return -1

def s2(k, R):
    if k==2:
        return int(R < 25)
    r_rem = 26 - R
    if p(k, r_rem) > 25/52:
        return int(r_rem > k - r_rem)
    return -1

if __name__ == '__main__':
    outcomes = play(5000, s1)
    print(np.mean(outcomes[:,0]))
    print(np.mean(outcomes[:,1]))
    print(np.mean(outcomes[:,2]))
