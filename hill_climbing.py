import numpy as np
import genetic_algorithm as ga

def cost_change(n1, n2, n3, n4):
    return n1.distance(n3) + n2.distance(n4) - n1.distance(n2) - n1.distance(n4)

def cost_change_3opt (best, n1, n2, n3, temp):
    combo_1 = best[:n1[0] + 1] + best[n1[1]:n2[0] + 1] + best[n2[1]: n3[0] + 1] + best[n3[1]:]
    combo_2 = best[:n1[0] + 1] + best[n1[1]:n2[0] + 1] + best[n3[0]: n2[1] - 1: -1] + best[n3[1]:]
    combo_3 = best[:n1[0] + 1] + best[n2[0]:n1[1] - 1: -1] + best[n2[1]: n3[0] + 1] + best[n3[1]:]
    combo_4 = best[:n1[0] + 1] + best[n2[0]:n1[1] - 1: -1] + best[n3[0]: n2[1] - 1: -1] + best[n3[1]:]
    combo_5 = best[:n1[0] + 1] + best[n2[1]: n3[0] + 1] + best[n1[1]:n2[0] + 1] + best[n3[1]:]
    combo_6 = best[:n1[0] + 1] + best[n2[1]: n3[0] + 1] + best[n2[0]:n1[1] - 1: -1] + best[n3[1]:]
    combo_7 = best[:n1[0] + 1] + best[n3[0]: n2[1] - 1: -1] + best[n1[1]:n2[0] + 1] + best[n3[1]:]
    combo_8 = best[:n1[0] + 1] + best[n3[0]: n2[1] - 1: -1] + best[n2[0]:n1[1] - 1: -1] + best[n3[1]:]
    combinations_array = [combo_1, combo_2, combo_3, combo_4, combo_5, combo_6, combo_7, combo_8]
    distances_array = list(map(lambda x: ga.Fitness(x).routeFitness(), combinations_array))
    min_distance = int(np.argmin(distances_array))
    difference = distances_array[min_distance] - ga.Fitness(combo_1).routeFitness()
    improved = False
    if ga.Fitness(temp).routeFitness() > distances_array[min_distance] and difference < 0 :
        improved = True
    return improved, combinations_array[min_distance], distances_array[min_distance]

def two_opt(route):
    best = route
    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route)):
            if j - i == 1: continue
            if cost_change(best[i - 1], best[i], best[j - 1], best[j]) < 0:
                best[i:j] = best[j - 1:i - 1:-1]
        route = best
    return best

def three_opt(route):
    best = route
    temp = best
    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route)):
            if j - i == 1: continue
            for k in range (j+1, len(route)):
                if k - j == 1: continue
                bestCombination = cost_change_3opt(route, [i,i+1], [j,j+1], [k,k+1], temp)
                if bestCombination[0]:
                    temp = bestCombination[1]
            if(ga.Fitness(best).routeFitness() > ga.Fitness(temp).routeFitness()):
                best = temp
    return best