import numpy as np
import genetic_algorithm as ga

def cost_change(n1, n2, n3, n4):
    return n1.distance(n3) + n2.distance(n4) - n1.distance(n2) - n1.distance(n4)


def two_opt(route):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                if cost_change(best[i - 1], best[i], best[j - 1], best[j]) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
        route = best
    return best