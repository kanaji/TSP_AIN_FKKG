import matplotlib.pyplot as plt
import numpy as np
import operator
import pandas as pd
import random
import hill_climbing as hc


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


def createRoute(cityList, seed):
    random.seed(seed)
    route = random.sample(cityList, len(cityList))
    return route


def initialPopulation(popSize, cityList, seed):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList, seed))
    return population


def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


def averageRoute(population):
    fitness_average = 0
    for i in range(0, len(population)):
        fitness_average += Fitness(population[i]).routeFitness()
    fitness_average = fitness_average / len(population)
    return fitness_average


def selection(popRanked, eliteSize, selectionType, size, seed):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()
    random.seed(seed)
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    if selectionType == 'Tournament':
        for i in range(0, len(popRanked) - eliteSize):
            winner = tournamentSelection(popRanked, size, seed)
            selectionResults.append(winner[0])
    if selectionType == 'Rank':
        for i in range(0, len(popRanked) - eliteSize):
            pick = 100 * random.random()
            for i in range(0, len(popRanked)):
                if pick <= df.iat[i, 3]:
                    selectionResults.append(popRanked[i][0])
                    break
    return selectionResults


def tournamentSelection(popRanked, size, seed):
    random.seed(seed)
    parents = random.choices(popRanked, k=size)
    parents = sorted(parents, reverse=True)
    return parents[0]


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2, seed, crossover_type, crossover_prob):
    random.seed(seed)
    # TODO: ADD CROSSOVERS
    if random.random() < crossover_prob:
        if crossover_type == "OX":
            # print("TEST OX")
            if random.random() <= 0.5:
                return parent1
            else:
                return parent2
        elif crossover_type == "CX":
            # print("TEST CX")
            if random.random() <= 0.5:
                return parent1
            else:
                return parent2
        elif crossover_type == "SCX":
            # print("TEST SCX")
            child = []
            childP1 = []
            childP2 = []
            geneA = int(random.random() * len(parent1))
            geneB = int(random.random() * len(parent1))

            startGene = min(geneA, geneB)
            endGene = max(geneA, geneB)

            for i in range(startGene, endGene):
                childP1.append(parent1[i])

            childP2 = [item for item in parent2 if item not in childP1]

            child = childP1 + childP2
            return child
    else:
        if random.random() <= 0.5:
            return parent1
        else:
            return parent2


def breedPopulation(matingpool, eliteSize, seed, crossover_type, crossover_prob):
    children = []
    length = len(matingpool) - eliteSize
    random.seed(seed)
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1], seed, crossover_type, crossover_prob)
        children.append(child)
    return children


def mutate_2_swap(individual, mutationRate, seed):
    random.seed(seed)
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutate_inversion(individual, mutationRate, seed):
    random.seed(seed)
    for city_1 in range(len(individual)):
        if random.random() < mutationRate:
            city_2 = int(random.random() * len(individual))

            if city_1 < city_2:
                begin = city_1
                end = city_2
            elif city_1 > city_2:
                begin = city_2
                end = city_1
            else:
                continue

            individual[begin:end] = individual[begin:end][::-1]
    return individual


def mutate_last_but_not_least(individual, mutationRate, seed):
    random.seed(seed)
    for city_1 in range(len(individual)):
        if random.random() < mutationRate:
            individual[0], individual[-1] = individual[-1], individual[0]
    return individual


def mutatePopulation(population, mutation_type, mutationRate, seed):
    mutatedPop = []
    if mutation_type == "2-swap":
        for ind in range(0, len(population)):
            mutatedInd = mutate_2_swap(population[ind], mutationRate, seed)
            mutatedPop.append(mutatedInd)
    elif mutation_type == "Inversion":
        for ind in range(0, len(population)):
            mutatedInd = mutate_inversion(population[ind], mutationRate, seed)
            mutatedPop.append(mutatedInd)
    elif mutation_type == "last_but_not_least":
        for ind in range(0, len(population)):
            mutatedInd = mutate_last_but_not_least(population[ind], mutationRate, seed)
            mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, hillclimb_type, hillclimb_generation, selection_type, selection_size,
                   crossover_type,
                   crossover_prob, mutation_type, mutation_prob, seed, current_gen):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize, selection_type, selection_size, seed)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize, seed, crossover_type, crossover_prob)
    mutatedPopulation = mutatePopulation(children, mutation_type, mutation_prob, seed)
    nextGeneration = hillClimbing(mutatedPopulation, hillclimb_type, hillclimb_generation, current_gen)

    return nextGeneration


def hillClimbing(population, hillclimb_type, hillclimb_generation, current_gen):
    if hillclimb_generation < ++current_gen:
        sortedPop = population
        sortedPop.sort(key=lambda x: Fitness(x).routeFitness(), reverse=True)
        if hillclimb_type == '2-opt':
            best = hc.two_opt(sortedPop[0])
            for route in population:
                if sortedPop[0] == route:
                    route = best
        if hillclimb_type == '3-opt':
            best = hc.three_opt(sortedPop[0])
            for route in population:
                if sortedPop[0] == route:
                    route = best
    return population


def geneticAlgorithm(population, popSize, generations, eliteSize, hillclimb_type, hillclimb_generation, selection_type,
                     selection_size, crossover_type,
                     crossover_prob, mutation_type, mutation_prob, seed):
    pop = initialPopulation(popSize, population, seed)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    routes = []
    routes.append(pop[rankRoutes(pop)[0][0]])

    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    average = []
    average.append(1 / averageRoute(pop))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, hillclimb_type, hillclimb_generation, selection_type, selection_size,
                             crossover_type,
                             crossover_prob, mutation_type, mutation_prob, seed, i)
        routes.append(pop[rankRoutes(pop)[0][0]])
        progress.append(1 / rankRoutes(pop)[0][1])
        average.append(1 / averageRoute(pop))

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute, progress, average, routes
