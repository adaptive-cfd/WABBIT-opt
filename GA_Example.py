"""
genetischer Algorithmus
https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
"""

import numpy
import ga
from time import process_time

t1_start = process_time()


# Number of parameters to be optimized
num_params = 7

# Inputs of the equation.
goal = numpy.random.randint(20, size=[1,num_params])
print("Goal : ", goal)

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 10
num_parents_mating = 2
# Defining the population size.
pop_size = (sol_per_pop,num_params) # The population will have sol_per_pop chromosome where each chromosome has num_params genes.
#Creating the initial population.
new_population = numpy.random.randint(10, size=pop_size)
print("first population : \n", new_population)

#num_generations = 5
#for generation in range(num_generations):
generation = 0
best_result = 100
while best_result >= 1:
    
    # Measuring the fitness of each chromosome in the population.
    fitness = ga.cal_pop_fitness(goal, new_population)

    # Selecting the best parents in the population for mating.
    parents = ga.select_mating_pool(new_population, fitness, num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = ga.crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], num_params))

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = ga.mutation(offspring_crossover, num_params)
    
    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
    
    best_result = numpy.min(numpy.sum(numpy.absolute(new_population - goal), axis=1))
    # The best result in the current iteration.
    
    generation = generation +1
    if generation % 1000 == 0:
        print("difference to goal (after another 1000 generations): ", best_result)

# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = ga.cal_pop_fitness(goal, new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.min(fitness))
print("Total number of generations : ", generation)
print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])

t1_stop = process_time()
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)