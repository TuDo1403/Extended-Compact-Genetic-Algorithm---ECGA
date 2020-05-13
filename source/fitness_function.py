import numpy as np

class FuncInf:
    NAME = ""
    F_FUNC = None

    def __init__(self, name, func):
        self.NAME = name
        self.F_FUNC = func

def onemax(ind):
    return sum(ind)

def trap(ind, k):
    f = 0
    for i in range(0, len(ind), k):
        u = np.sum(ind[i : i + k])
        f += u if u == k else k - u - 1
    return f

def trap_five(ind):
    return trap(ind, 5)

def trap_four(ind):
    return trap(ind, 4)