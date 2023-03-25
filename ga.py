import numpy as np
import random
from main import predict_anomalies

POPULATION_SIZE = 100
GENERATIONS = 500
MUTATION_RATE = 0.1
ELITISM_RATE = 0.1

SEQUENCE_LENGTH = 10  # The same sequence length used when training the model


# Initialize the population
def init_population():
    return [
        np.random.randint(low=0, high=127, size=(SEQUENCE_LENGTH, 3))
        for _ in range(POPULATION_SIZE)
    ]


# Fitness function (the number of detected anomalies for each individual)
def fitness(individual, model):
    num_anomalies = predict_anomalies(model, individual)
    return num_anomalies


# Selection function (roulette wheel selection)
def selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    selected_idx = np.random.choice(
        np.arange(len(population)), size=len(population), p=fitnesses / total_fitness
    )
    return [population[i] for i in selected_idx]


# Crossover function (uniform crossover)
def crossover(parent1, parent2):
    child = np.copy(parent1)
    for i in range(SEQUENCE_LENGTH):
        for j in range(3):
            if random.random() < 0.5:
                child[i, j] = parent2[i, j]
    return child


# Mutation function (randomly change a note, velocity, or time value)
def mutate(individual):
    for i in range(SEQUENCE_LENGTH):
        for j in range(3):
            if random.random() < MUTATION_RATE:
                individual[i, j] = np.random.randint(0, 127)


# Main GA loop
population = init_population()

for generation in range(GENERATIONS):
    # Evaluate the fitness of each individual
    fitnesses = [fitness(individual, model) for individual in population]

    # Sort the population by fitness (descending)
    sorted_indices = np.argsort(fitnesses)[::-1]
    population = [population[i] for i in sorted_indices]
    fitnesses = [fitnesses[i] for i in sorted_indices]

    # Elitism: keep the top individuals
    num_elites = int(POPULATION_SIZE * ELITISM_RATE)
    next_population = population[:num_elites]

    # Selection, crossover, and mutation
    selected_population = selection(population, fitnesses)
    while len(next_population) < POPULATION_SIZE:
        parent1, parent2 = random.sample(selected_population, 2)
        child = crossover(parent1, parent2)
        mutate(child)
        next_population.append(child)

    population = next_population

    # Print the best individual's fitness
    print(f"Generation {generation + 1}, Best fitness: {fitnesses[0]}")

# The best individual
best_individual = population[0]
