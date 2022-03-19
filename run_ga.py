import time
import numpy as np
import pygad
from multiprocessing import Pool

class Solution:
    def __init__(self, allocations, people, constraints, from_date, to_date):
        self.allocations = allocations
        self.people = people
        self.constraints = constraints
        self.from_date = from_date
        self.to_date = to_date
        self.people_by_id = {p.id: p for p in self.people}
        self.hours_by_day = {}
        current_day = list(self.allocations.keys())[0].date()
        hours = []
        for hour in self.allocations.keys():
            if hour.date() == current_day:
                hours.append(hour)
            else:
                self.hours_by_day[current_day] = hours
                hours = [hour]
                current_day = hour.date()
        self.hours_by_day[current_day] = hours
                
    def people_scheduled_for_hour(self, hour):
        return [self.people_by_id[self.allocations[hour]]]

    def calculate_contiguousness_bonus(self):
        # 50 points for one person per day
        score = 0
        for hours in self.hours_by_day.values():
            people = set()
            for hour in hours:
                people.update(self.people_scheduled_for_hour(hour))
            diff = len(self.people) - len(people)
            score += {
                4: 50,
                3: 20,
                2: 5,
                1: 1,
                0: 0
            }[diff]
        if score == 5 * 50:
            raise RuntimeError()
        return score


class FitnessWrapper:
    def __init__(self, hours_to_schedule, people, constraints, from_date, to_date):
        self.hours_to_schedule = hours_to_schedule
        self.people = people
        self.constraints = constraints
        self.from_date = from_date
        self.to_date = to_date

    @property
    def __code__(self):
        code = lambda: None
        code.co_argcount = 2  # Work around an annoying aspect of pygad
        return code

    def __call__(self, x):
        params, index = x
        value = self.evaluate_fitness(self.generate_solution(params))
        return value, index

    def evaluate_fitness(self, solution):
        score = 0
        for constraint in self.constraints:
            if constraint.is_satisfied_by_solution(solution):
                score += constraint.value
        score += solution.calculate_contiguousness_bonus()
        return score

    def generate_solution(self, params):
        return Solution(dict(zip(self.hours_to_schedule, params)), self.people, self.constraints, self.from_date, self.to_date)


class PooledGA(pygad.GA):
    def __init__(self, *args, pool=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = pool

    def callback_gen(self):
        print(time.time(), "Generation : ", self.generations_completed)
        print(time.time(), "Fitness of the best solution :", self.best_solution()[1])

    def cal_pop_fitness(self):
        sols_to_calc = []
        precalced_sols = {}
        for sol_idx, sol in enumerate(self.population):
            # Check if the parent's fitness value is already calculated. If so, use it instead of calling the fitness function.
            if not (self.last_generation_parents is None) and len(np.where(np.all(self.last_generation_parents == sol, axis=1))[0] > 0):
                # Index of the parent in the parents array (self.last_generation_parents). This is not its index within the population.
                parent_idx = np.where(np.all(self.last_generation_parents == sol, axis=1))[0][0]
                # Index of the parent in the population.
                parent_idx = self.last_generation_parents_indices[parent_idx]
                # Use the parent's index to return its pre-calculated fitness value.
                fitness = self.last_generation_fitness[parent_idx]
                precalced_sols[sol_idx] = fitness
            else:
                sols_to_calc.append((sol, sol_idx))

        calced_pop_fitness = self.pool.map(self.fitness_func, sols_to_calc)
        for fitness, idx in calced_pop_fitness:
            precalced_sols[idx] = fitness

        pop_fitness = np.array([precalced_sols[idx] for idx in sorted(precalced_sols.keys())])
        return pop_fitness


def run_ga(options):
    fitness_func = FitnessWrapper(
        people=options['people'],
        constraints=options['constraints'],
        hours_to_schedule=options['hours_to_schedule'],
        from_date=options['from'],
        to_date=options['to'])

    ids = [p.id for p in options['people']]
    gene_space = [
        ids for hour in options['hours_to_schedule']
    ]
    num_genes = len(gene_space)
    with Pool() as pool:
        ga_instance =PooledGA(num_generations=options['num_generations'],
                              num_parents_mating=options['num_parents_mating'],
                              fitness_func=fitness_func,
                              sol_per_pop=options['sol_per_pop'],
                              mutation_percent_genes=options['mutation_percent_genes'],
                              num_genes=num_genes,
                              on_generation=PooledGA.callback_gen,
                              pool=pool,
                              gene_space=gene_space,
                              parent_selection_type=options['parent_selection_type'],
                              keep_parents=options['keep_parents'],
                              crossover_type=options['crossover_type'],
                              mutation_type=options['mutation_type'],
                              mutation_probability=options['mutation_probability'],
                              save_best_solutions=True)
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print(solution, solution_fitness, solution_idx, [str(s) for s in solution])
        for sol, fitn in zip(ga_instance.best_solutions, ga_instance.best_solutions_fitness):
            print([str(x) for x in sol], fitn)
