from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

class Classification:
    def __init__(self):
        pass
    
    def auto_classify(self, type_method, param_grid=None):
        if type_method == 'logistic':
            return self.logistic(param_grid)
        elif type_method == 'svm':
            return self.svm(param_grid)
        elif type_method == 'decisiontree':
            return self.decision_tree(param_grid)
        elif type_method == 'randomforest':
            return self.random_forest(param_grid)
        elif type_method == 'knn':
            return self.knn(param_grid)
        elif type_method == 'mlp':
            return self.mlp(param_grid)
        elif type_method == 'adaboost':
            return self.adaboost(param_grid)
        elif type_method == 'gradientboosting':
            return self.gradient_boosting(param_grid)
        elif type_method == 'gaussianprocess':
            return self.gaussian_process(param_grid)
        else:
            raise ValueError('Invalid method')

    def logistic(self, param_grid=None):
        model = LogisticRegression()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def svm(self, param_grid=None):
        model = SVC()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def decision_tree(self, param_grid=None):
        model = DecisionTreeClassifier()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def random_forest(self, param_grid=None):
        model = RandomForestClassifier()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def knn(self, param_grid=None):
        model = KNeighborsClassifier()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def mlp(self, param_grid=None):
        model = MLPClassifier()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def adaboost(self, param_grid=None):
        model = AdaBoostClassifier()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def gradient_boosting(self, param_grid=None):
        model = GradientBoostingClassifier()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

    def gaussian_process(self, param_grid=None):
        model = GaussianNB()
        if param_grid:
            model = GridSearchCV(model, param_grid)
        return model

# # Example usage:
# classification = Classification()
# param_grid = {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}
# model = classification.auto_classify('svm', param_grid)
