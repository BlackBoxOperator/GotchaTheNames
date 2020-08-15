#!/bin/python

import array
import random

import numpy as np

from pprint import pprint
from deap import algorithms
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap import cma

from fit import fitness
from oracle import keyword_tuner

random.seed(64)
np.random.seed(64)

class WeightCMAES():
    """
    Parameters
    ----------
    ind_size: size of the individuales
    problem: the problem to sovle

    Reference:
    ----------
    Deap: a evolutionary tool for solving optimization problem
    https://github.com/DEAP/deap
    """


    def __init__(self, ind_size, problem):
        self.ind_size = ind_size
        # creator
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("evaluate", problem)
        self.toolbox.register("generate", self.gen_new_pop, creator.Individual)
        self.toolbox.register("update", self.update_new_pop)

        # strategy
        self.strategy = cma.Strategy(centroid=[0.5]*self.ind_size, sigma=0.15, lambda_=30)

    def adjust(self, x):
        if x >= 1.0:
            return 1.0
        if x <= 0.0:
            return 0.0
        return x

    def update_new_pop(self, y):
        self.strategy.update(y)
        for group in y:
            group[:] = [self.adjust(x) for x in group]


    def gen_new_pop(self, y):
        pop = self.strategy.generate(y)
        for group in pop:
            group[:] = [self.adjust(x) for x in group]
        return pop


    def run(self):
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop, logbook = algorithms.eaGenerateUpdate(
            self.toolbox, ngen=200, stats=stats, halloffame=hof)
        return pop, logbook, hof

def problem(keys, kt):
    """
    return a function that return from kt
    """
    def return_function(x):
        table = dict(zip(keys, x))
        acc, vacc, xacc = kt(table)
        return [(vacc + xacc)/2]

    return return_function

def main():
    # initialize
    with open("../scripts/keyword/key_tfidf.txt") as key_file:
        table = {k:0.5 for k in key_file.read().split()}

    kt = keyword_tuner(table, "../query/tquery.csv")

    keys = [k for k in table]
    weights = [table[k] for k in table]
    # table = dict(zip(keys, weights))


    # update table
    acc, vacc, xacc = kt(table)

    # f = lambda x : [kt(dict(zip(keys, x)))[1]]
    # create a problem
    f = problem(keys, kt)

    print(f(weights))

    ind_size = len(table)
    cmaes = WeightCMAES(ind_size, f)
    pop, log, hof = cmaes.run()

    logbook = open('logbook.txt', 'w')
    key_result = open('keyResult.dict', 'w')

    print(log)
    print(log, file = logbook)

    for h in hof:
        print("individual: ", [x for x in h], " value: ", h.fitness.values)
        print("individual: ", [x for x in h], " value: ", h.fitness.values, file = logbook)


    pprint(dict(zip(keys, hof[0])), stream = key_result)
    logbook.close()
    key_result.close()


if __name__ == "__main__":
    main()
