import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Genetic_Algorithm:
    def __init__(self, model, X_train, y_train, X_test, y_test, population_size=10, generations=10, mutation_rate=0.1, type_method='classification'):
        self.model = model
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
            population.append(individual)
        return np.array(population)
    
    def calculate_fitness(self):
        fitness = []
        for individual in self.population:
            if self.type_method == 'classification':
                self.model.fit(self.X_train[:, individual==1], self.y_train)
                fitness.append(self.model.score(self.X_test[:, individual==1], self.y_test))
            elif self.type_method == 'regression':
                self.model.fit(self.X_train[:, individual==1], self.y_train)
                fitness.append(self.model.score(self.X_test[:, individual==1], self.y_test))
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
        return individual
    
    def evolve(self):
        new_population = []
        for i in range(self.population_size // 2):  # since we are generating two children at a time
            parent1 = self.selection()
            parent2 = self.selection()
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            new_population.extend([child1, child2])
        self.population = np.array(new_population)
        self.fitness = self.calculate_fitness()
        
        if np.max(self.fitness) > self.best_fitness:
            self.best_individual = self.population[np.argmax(self.fitness)]
            self.best_fitness = np.max(self.fitness)
            
    def run(self):
        for i in range(self.generations):
            self.evolve()
            print(f'Generation {i+1} - Best Fitness: {self.best_fitness}')
        return self.best_individual, self.best_fitness
    
if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_classifier = RandomForestClassifier(n_estimators=100)
    ga_classifier = Genetic_Algorithm(model_classifier, X_train, y_train, X_test, y_test, population_size=10, generations=10, mutation_rate=0.1, type_method='classification')
    best_individual_classifier, best_fitness_classifier = ga_classifier.run()
    
    model_regressor = RandomForestRegressor(n_estimators=100)
    ga_regressor = Genetic_Algorithm(model_regressor, X_train, y_train, X_test, y_test, population_size=10, generations=10, mutation_rate=0.1, type_method='regression')
    best_individual_regressor, best_fitness_regressor = ga_regressor.run()
    
    model_classifier.fit(X_train[:, best_individual_classifier==1], y_train)
    y_pred_classifier = model_classifier.predict(X_test[:, best_individual_classifier==1])
    accuracy = accuracy_score(y_test, y_pred_classifier)
    print(f'Accuracy: {accuracy}')
    
    model_regressor.fit(X_train[:, best_individual_regressor==1], y_train)
    y_pred_regressor = model_regressor.predict(X_test[:, best_individual_regressor==1])
    mse = mean_squared_error(y_test, y_pred_regressor)
    print(f'Mean Squared Error: {mse}')
    
    print(f'Best Individual (Classification): {best_individual_classifier}')
    print(f'Best Fitness (Classification): {best_fitness_classifier}')
    print(f'Number of Features (Classification): {np.sum(best_individual_classifier)}')
    print(f'Selected Features (Classification): {data.feature_names[best_individual_classifier==1]}')
    
    print(f'Best Individual (Regression): {best_individual_regressor}')
    print(f'Best Fitness (Regression): {best_fitness_regressor}')
    print(f'Number of Features (Regression): {np.sum(best_individual_regressor)}')
    print(f'Selected Features (Regression): {data.feature_names[best_individual_regressor==1]}')
