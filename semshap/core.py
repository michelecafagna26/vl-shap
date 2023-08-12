import random
import itertools
import scipy

def bouncing_range(n):
    """
    Generate a list of int form 0-n with alternate indexes. Ex: 0, n, 1, n-1, 2, n-2, ...
    """
    j = 0
    items = []
    for i in range(n + 1):

        if i % 2 == 0:
            items += [n - j]

        else:
            items += [j]
            j += 1

    return items


def powerset(iterable, sampling="range"):
    s = list(iterable)

    if sampling == "range":
        # generate the combinations ordered by size: from the smaller size to the largest
        iterable = range(len(s) + 1)

    elif sampling == "optimal":
        # generate the combinations by picking alternatively the largest and the smallest.
        # This is equivalent to select the high weight combinations first
        iterable = bouncing_range(len(s) + 1)

    elif sampling == "random":
        iterable = [i for i in range(len(s) + 1)]
        random.shuffle(iterable)

    else:
        raise ValueError(f"{sampling} is not recognized. Allowed modes are ['range', 'bounce', 'random']")

    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in iterable)


def shapley_kernel(M, s):
    if s == 0 or s == M:
        return 10000
    return (M - 1) / (scipy.special.binom(M, s) * s * (M - s))


