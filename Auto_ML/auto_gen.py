import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_linnerud
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from tqdm import tqdm

class Auto_GA_FS_Tr:
    def __init__(self, model, param_grid, X_train, y_train, X_test, y_test, population_size=10, generations=10, mutation_rate=0.1, type_method='classification'):
        self.model = model
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.type_method = type_method
        self.population = self.generate_population()
        self.fitness = self.calculate_fitness()
        self.best_individual = self.population[np.argmax(self.fitness)]
        self.best_fitness = np.max(self.fitness)
        
    def generate_population(self):
        population = []
        for i in range(self.population_size):
            individual = np.random.choice([0, 1], size=self.X_train.shape[1])
            while np.sum(individual) == 0:  # Ensure at least one feature is selected
                individual = np.random.choice([0, 1], size=self.X_train.shape[1])
            population.append(individual)
        return np.array(population)
    
    def calculate_fitness(self):
        fitness = []
        for individual in self.population:
            if np.sum(individual) == 0:  # Skip if no features are selected
                fitness.append(float('-inf'))
            else:
                if self.type_method == 'classification':
                    self.model.fit(self.X_train[:, individual==1], self.y_train)
                    fitness.append(self.model.score(self.X_test[:, individual==1], self.y_test))
                elif self.type_method == 'regression':
                    # Hyperparameter tuning
                    grid_search = GridSearchCV(self.model, self.param_grid, scoring='r2', cv=5)
                    grid_search.fit(self.X_train[:, individual==1], self.y_train)
                    best_model = grid_search.best_estimator_
                    fitness.append(best_model.score(self.X_test[:, individual==1], self.y_test))
                elif self.type_method == 'clustering':
                    self.model.fit(self.X_train[:, individual==1])
                    fitness.append(self.model.score(self.X_test[:, individual==1]))
        return np.array(fitness)
    
    def selection(self):
        idx = np.random.choice(range(self.population_size), size=2, replace=False)
        return self.population[idx[np.argmax(self.fitness[idx])]]
    
    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(0, len(parent1))
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2
    
    def mutation(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        if np.sum(individual) == 0:  # Ensure at least one feature is selected after mutation
            individual[np.random.randint(len(individual))] = 1
        return individual
    
    def evolve(self):
        new_population = []
        for i in range(self.population_size // 2):
            parent1 = self.selection()
            parent2 = self.selection()
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            new_population.extend([child1, child2])
            
        self.population = np.array(new_population)
        self.fitness = self.calculate_fitness()
        self.best_individual = self.population[np.argmax(self.fitness)]
        self.best_fitness = np.max(self.fitness)
        return self.best_individual, self.best_fitness
    
    def fit(self):
        for i in tqdm(range(self.generations), desc='Generations'):
            best_individual, best_fitness = self.evolve()
            tqdm.write(f'Generation {i+1} - Best Fitness: {best_fitness}')
        return self.best_individual, self.best_fitness
    
    def transform(self, X):
        return X[:, self.best_individual==1]
    
    def fit_transform(self):
        self.fit()
        return self.transform(self.X_train), self.transform(self.X_test)

