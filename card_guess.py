import numpy as np
from scipy.misc import comb
from numpy import random
import matplotlib.pyplot as plt

def shuffle(n, one_riffle=False):
    '''
    -Inputs:
    int n: the number of cards in the deck to be shuffled.
    Inital arrangement of cards is taken to be (1,2,3,...,n) where 1 is the top
    card.
    bool one_riffle: if set to true it shuffles the cards according to one run
    of the GSR riffle shuffle model.
    -Output:
    A list consisting of the numbers 1 to n shuffled.
    Idea for improvement: popping from beginning of list is slow.
    '''
    deck = list(range(1,n+1))
    new_deck = []
    if one_riffle:
        # One riffle shuffle
        c = list(range(1, n+1))
        prob_c = [comb(n, x)/(2**n) for x in c]
        cut = random.choice(c, p=prob_c)
        left = deck[:cut]
        right = deck[cut:]
        for i in range(n):
            p_left = len(left)/(n-i)
            if random.ranf() <= p_left:
                new_deck.append(left[0])
                del(left[0])
            else:
                new_deck.append(right[0])
                del(right[0])
    else:
        # Random shuffle
        for i in range(n, 0, -1):
            j = random.randint(0, i)
            new_deck.append(deck[j])
            del(deck[j])
    return new_deck


def run_sims(deck_size, num_sims, strategy, one_riffle=False, feedback=False):
    '''
    - Inputs:
    int deck_size: size of deck.
    int num_sims: number of simulated games to play.
    function int strategy(int deck_size, int turn,  list of ints correct_guesses,
    list of ints cards_seen): a function which returns a guess for which card is
    on top of a deck of size deck_size, at turn turn (0 indexed), given a list
    of correct guesses so far, and, if feedback=True, a list of cards seen so
    far.
    bool one_riffle: if set to true shuffles the cards by GSR riffle shuffling
    them once.
    bool feedback: if set to true gives feedback to strategy.
    -Outputs:
    list of num of correct guesses in each game.
    '''
    correct = []
    for i in range(num_sims):
        deck = shuffle(deck_size, one_riffle)
        correct_guesses = []
        cards_seen = []
        for j in range(deck_size):
            if feedback:
                guess = strategy(deck_size, j, correct_guesses, cards_seen)
            else:
                guess = strategy(deck_size, j, correct_guesses)
            if guess == deck[j]:
                correct_guesses.append(guess)
            cards_seen.append(deck[j])
        correct.append(len(correct_guesses))
    return correct

def no_feedback(n, turn, correct_guesses):
    '''
    A strategy for card guessing with no feedback. Basically, keep guessing 1
    until you get it, then guess 2 and so on.
    '''
    return len(correct_guesses) + 1

def feedback(n, turn, correct_guesses, cards_seen):
    '''
    A strategy for card guessing with feedback. It works as follows:
    Start by guessing 1. Continue guessing the increasing sequence (1,2,3...)
    until you get a card wrong.
    When you do, the card revealed will be exactly 1 above the cutoff.
    Once you determine the cutoff you will be able to determine the left and
    right stacks from which the deck was riffled. Cross off the cards seen off
    of each stack. Guess the top card of whichever stack is deeper at the
    moment of the guess.
    Note: It would be better to implement guessing strategies as classes instead
    of functions. This would avoid all the recomputation we're having to do with
    this design. But oh well. Sunk costs...
    '''
    for c in cards_seen:
        if c > turn:
            cutoff = c - 1
            break
    else:
        cutoff = -1
    deck = list(range(1, n+1))
    if cutoff > 0:
        left = deck[:cutoff]
        right = deck[cutoff:]
        for c in cards_seen:
            if c in left: left.remove(c)
            else: right.remove(c)
        if len(left) >= len(right):
            return left[0]
        else:
            return right[0]
    else:
        return turn + 1

if __name__ == '__main__':
    results = run_sims(52, 10000, feedback, True, True)
    plt.figure()
    plt.hist(results, bins=list(range(min(results),max(results)+2)),
        label='Mean={}, Std={}, Mean acc={}'.format(round(np.mean(results),2),
            round(np.std(results),2), round(np.mean(results)/52, 3)),
        alpha=0.75, color='red', ec='black')
    plt.legend()
    plt.show()
