import numpy as np
import fitness_function as ff
import math

def get_MDL(pop, model):
    """ Compute minimum description length
    """
    return get_CPC(pop, model) + get_MC(pop, model)


def get_CPC(pop, model):
    S = np.array(list(map(len, model)))
    dec_events = [np.arange(2**s, dtype=np.uint8)[:, np.newaxis] for s in S]
    bin_events = [np.unpackbits(de, axis=1) for de in dec_events]
    num_groups = len(model)
    events = [bin_events[i][:, -S[i]:] for i in range(num_groups)]
    
    sum_entropy = 0
    N = len(pop)
    universal_set = N + 1
    for i in range(num_groups):
        for event in events[i]:
            group = pop[:, model[i]]
            prop = count_events(event, group) / universal_set
            if prop != 0:
                sum_entropy += prop * np.log2(1/prop)

            
    return N * sum_entropy

def count_events(event, group):
    count = 0
    for record in group:
        if (event == record).all():
            count += 1

    return count


def get_MC(pop, model):
    n = len(pop)
    s = np.array(list(map(len, model)))
    return np.log2(n + 1) * np.sum(2**s - 1)

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

    population = initialize_population(user_config.NUM_INDS, user_config.NUM_PARAMS)
    pop_fitness = evaluate(population, func_inf.F_FUNC)
    num_eval_func_calls = len(pop_fitness)

    pop_model = [[group] for group in np.arange(user_config.NUM_PARAMS)]

    selection_size = len(population)
    

    # print("#Gen 0:")
    # print(pop_fitness)
    generation = 0
    while len(np.unique(pop_fitness)) != 1:
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
        
        # print("#Gen {}:".format(generation))
        # print(model)
        # print(pop_fitness)

    # print("#Result:")
    # print(population)
    # print(pop_fitness)
        
    optimized_solution_found = user_config.NUM_PARAMS == np.unique(pop_fitness)
    print(optimized_solution_found[0], num_eval_func_calls)
    return (optimized_solution_found[0], num_eval_func_calls)

def converge(current_model, new_model):
    for item in current_model:
        if item not in new_model:
            return False
    return True


class ECGAConfig:
    NUM_PARAMS = 4
    NUM_INDS = 4
    TOURNAMENT_SIZE = 4

    def __init__(self, num_inds, num_params, tournament_size=4):
        self.NUM_PARAMS = num_params
        self.NUM_INDS = num_inds
        self.TOURNAMENT_SIZE = tournament_size

seed = 1
user_config = ECGAConfig(20, 4, 4)
func_inf = ff.FuncInf('Trap Four', ff.trap_four)

print(ECGA(user_config, func_inf))
