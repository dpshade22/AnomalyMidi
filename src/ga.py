import pandas as pd
import numpy as np
import random
from copy import deepcopy
from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("anomalous/elise_modified01.csv")


# Your actual model should be used here
model = load_model("model/model.h5")

# Your predict_anomalies function
def predict_anomalies(model, df, sequence_length=500):
    df = df[["note", "velocity", "time"]].copy()
    scaler = MinMaxScaler()
    df.loc[:, ["note", "velocity", "time"]] = scaler.fit_transform(
        df[["note", "velocity", "time"]]
    )

    if len(df) < sequence_length:
        padding = pd.DataFrame(
            np.zeros((sequence_length - len(df), 3)),
            columns=["note", "velocity", "time"],
        )
        df = pd.concat([df, padding], ignore_index=True)

    input_data = df.iloc[:sequence_length, :].values.reshape(1, sequence_length, -1)
    predictions = model.predict(input_data)
    predictions[predictions < 0] = 0
    return int(np.round(np.sum(predictions)))


# Genetic Algorithm Parameters
population_size = 50
generations = 100
mutation_rate = 0.1
elite_size = 5

# Helper functions
def create_individual():
    individual = data.copy()
    for col in individual.columns:
        individual[col] = individual[col].apply(lambda x: x + random.randint(-10, 10))
    return individual


def create_population():
    return [create_individual() for _ in range(population_size)]


def fitness_function(individual):
    return predict_anomalies(model, individual)


def select_parents(population, fitnesses):
    sorted_indices = np.argsort(fitnesses)
    selected_parents = [population[i] for i in sorted_indices[:elite_size]]
    return selected_parents


def crossover(parent1, parent2):
    child = parent1.copy()
    crossover_point = random.randint(0, len(parent1))
    for col in child.columns:
        child[col].iloc[:crossover_point] = parent2[col].iloc[:crossover_point]
    return child


def mutate(individual):
    for col in individual.columns:
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[col].iloc[i] += random.randint(-10, 10)
    return individual


# Genetic Algorithm Loop
population = create_population()
best_fitness_so_far = float("inf")

for generation in range(generations):
    fitnesses = [fitness_function(individual) for individual in population]

    current_best_fitness = min(fitnesses)
    if current_best_fitness < best_fitness_so_far:
        best_fitness_so_far = current_best_fitness
        print(f"Generation {generation}: Best fitness so far: {best_fitness_so_far}")

    parents = select_parents(population, fitnesses)
    offspring = [
        crossover(random.choice(parents), random.choice(parents))
        for _ in range(population_size - elite_size)
    ]
    offspring = [mutate(child) for child in offspring]
    population = parents + offspring

# Get the best solution
best_solution = min(population, key=fitness_function)
best_fitness = fitness_function(best_solution)
best_solution.to_csv("best_solution.csv", index=False)

print("\nFinal best solution:", best_solution)
print("Final best fitness:", best_fitness)
