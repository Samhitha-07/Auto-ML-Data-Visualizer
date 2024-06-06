import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, RANSACRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

class Regression:
    def __init__(self):
        pass

    def auto_regressor(self, type_method, param_grid=None):
        if type_method == 'linear':
            return self.linear(param_grid)
        elif type_method == 'ridge':
            return self.ridge(param_grid)
        elif type_method == 'lasso':
            return self.lasso(param_grid)
        elif type_method == 'elasticnet':
            return self.elasticnet(param_grid)
        elif type_method == 'bayesianridge':
            return self.bayesianridge(param_grid)
        elif type_method == 'ransac':
            return self.ransac(param_grid)
        elif type_method == 'svr':
            return self.svr(param_grid)
        elif type_method == 'decisiontree':
            return self.decision_tree(param_grid)
        elif type_method == 'randomforest':
            return self.random_forest(param_grid)
        elif type_method == 'adaboost':
            return self.adaboost(param_grid)
        elif type_method == 'gradientboosting':
            return self.gradient_boosting(param_grid)
        elif type_method == 'mlp':
            return self.mlp(param_grid)
        elif type_method == 'knn':
            return self.knn(param_grid)
        elif type_method == 'gaussianprocess':
            return self.gaussian_process(param_grid)
        else:
            raise ValueError('Invalid method')

    def linear(self, param_grid=None):
        model = LinearRegression()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def ridge(self, param_grid=None):
        model = Ridge()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def lasso(self, param_grid=None):
        model = Lasso()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def elasticnet(self, param_grid=None):
        model = ElasticNet()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def bayesianridge(self, param_grid=None):
        model = BayesianRidge()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def ransac(self, param_grid=None):
        model = RANSACRegressor()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def svr(self, param_grid=None):
        if not param_grid:
            param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
        model = SVR()
        model = GridSearchCV(model, param_grid)
        return model

    def decision_tree(self, param_grid=None):
        model = DecisionTreeRegressor()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def random_forest(self, param_grid=None):
        model = RandomForestRegressor()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def adaboost(self, param_grid=None):
        model = AdaBoostRegressor()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def gradient_boosting(self, param_grid=None):
        model = GradientBoostingRegressor()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def mlp(self, param_grid=None):
        model = MLPRegressor()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def knn(self, param_grid=None):
        model = KNeighborsRegressor()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def gaussian_process(self, param_grid=None):
        model = GaussianProcessRegressor()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

# # Example usage:
# regression = Regression()
# param_grid = {'alpha': [0.1, 1.0, 10.0]}
# model = regression.auto_regressor('ridge', param_grid)
