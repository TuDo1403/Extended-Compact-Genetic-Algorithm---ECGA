import numpy as np
import fitness_function as ff
import math

from numpy import arange
from numpy import newaxis
def get_MDL(pop, model):
    """ Compute minimum description length
    """
    # return get_CPC(pop, model) + get_MC(pop, model)
    N = len(pop)
    S = np.array(list(map(len, model)))
    MC = np.log2(N + 1) * np.sum(2**S - 1)

    entropy = 0
    num_groups = len(model)
    events = [arange(2**S[i])[:, newaxis] >> np.arange(S[i])[::-1] & 1 for i in range(num_groups)]
    for i in range(num_groups):
        for event in events[i]:
            group = pop[:, model[i]]
            match = np.sum(group == event, axis=1)
            prop = np.count_nonzero(match == len(event)) / (N+1)
            if prop != 0:
                entropy += prop * np.log2(1/prop)
            
    CPC = N * entropy
    return CPC + MC


def link_groups(model):
    models = []
    for i in range(len(model) - 1):
        for j in range(i+1, len(model)):
            new_group = model.copy()
            del new_group[i]
            del new_group[j - 1]
            new_group.append(model[i] + model[j])
            models.append(new_group)
    return models

def learn_marginal_product(pop, model):
    current_MDL = get_MDL(pop, model)
    new_models = link_groups(model)
    new_MDLs = np.array([get_MDL(pop, model) for model in new_models])
    return model if current_MDL < np.min(new_MDLs) else new_models[np.argmin(new_MDLs)]


def initialize_population(num_inds, num_params):
    return np.random.randint(low=0, high=2, size=(num_inds, num_params))

def evaluate(pop, func):
    return np.array(list(map(func, pop)))

def variate(pop, model):
    (num_inds, num_params) = np.shape(pop)
    indices = np.array(range(num_inds))

    offsprings = []
    np.random.shuffle(indices)

    for i in range(0, num_inds, 2):
        index1 = indices[i]
        index2 = indices[i+1]
        offspring1 = pop[index1].copy()
        offspring2 = pop[index2].copy()

        for group in model:
            if np.random.rand() < 0.5:
                offspring1[group], offspring2[group] = offspring2[group].copy(), offspring1[group]
            
        offsprings.append(offspring1)
        offsprings.append(offspring2)

    return np.reshape(offsprings, (num_inds, num_params))

def tournament_selection(pool_fitness, tournament_size, selection_size):
    num_individuals = len(pool_fitness)
    indices = np.array(range(num_individuals))
    selected_indices = []

    while len(selected_indices) < selection_size:
        np.random.shuffle(indices)

        for i in range(0, num_individuals, tournament_size):
            idx_tournament = indices[i:i+tournament_size]
            winner = list(filter(lambda x : pool_fitness[x] == max(pool_fitness[idx_tournament]), idx_tournament))
            selected_indices.append(np.random.choice(winner))

    return selected_indices



def ECGA(user_config, func_inf, seed_num=1):
    np.random.seed(seed_num)

    population = initialize_population(user_config.POP_SIZE, user_config.PROBLEM_SIZE)
    pop_fitness = evaluate(population, func_inf.F_FUNC)
    num_eval_func_calls = len(pop_fitness)

    pop_model = [[group] for group in np.arange(user_config.PROBLEM_SIZE)]

    selection_size = len(population)
    

    # print("#Gen 0:")
    # print(pop_fitness)
    generation = 0
    while not pop_converge(population):
        while len(pop_model) != 1:
            model = learn_marginal_product(population, pop_model)
            if converge(model, pop_model):
                break
            else:
                pop_model = model

        offsprings = variate(population, pop_model)
        off_fitness = evaluate(offsprings, func_inf.F_FUNC)    
        num_eval_func_calls += len(off_fitness)

        pool = np.vstack((population, offsprings))
        pool_fitness = np.hstack((pop_fitness, off_fitness))

        pool_indices = tournament_selection(pool_fitness, user_config.TOURNAMENT_SIZE, selection_size)
        population = pool[pool_indices]
        pop_fitness = pool_fitness[pool_indices]

        generation += 1
        
        print("#Gen {}:".format(generation))
        print(model)
        # print(pop_fitness)

    # print("#Result:")
    # print(population)
    # print(pop_fitness)
        
    optimized_solution_found = user_config.PROBLEM_SIZE == np.unique(pop_fitness)
    print(population)
    return (optimized_solution_found[0], num_eval_func_calls)

def converge(current_model, new_model):
    for item in current_model:
        if item not in new_model:
            return False
    return True

def pop_converge(pop):
    return len(np.unique(pop, axis=0)) == 1


class ECGAConfig:
    NAME = 'ECGA'
    POP_SIZE = 4
    PROBLEM_SIZE = 4
    TOURNAMENT_SIZE = 4

    def __init__(self, pop_size, problem_size, tournament_size=4):
        self.PROBLEM_SIZE = problem_size
        self.POP_SIZE = pop_size
        self.TOURNAMENT_SIZE = tournament_size

seed = 1
user_config = ECGAConfig(1000, 20, 4)
func_inf = ff.FuncInf('Trap Five', ff.trap_five)

print(ECGA(user_config, func_inf))
