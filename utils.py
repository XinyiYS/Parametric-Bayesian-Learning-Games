from itertools import combinations
def powerset(L):
    n = len(L)
    P_set = []
    for i in range(0,n+1):
        for element in combinations(L, i):
            P_set.append(list(element))
    return P_set

