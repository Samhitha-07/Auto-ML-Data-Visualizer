
# ------- AUTO ML -----
#      Preprocessing
# ----------------------

# @ Author: SEM 4 GROUP 9

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular


class Visualization:
    
    def __init__(self):
        pass
    
    def plot_correlation(self, data):
        """
        Plot Correlation Matrix
        """
        corr = data.corr()
        plt.figure(figsize=(10, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
        
    def plot_histogram(self, data):
        """
        Plot Histogram
        """
        data.hist(figsize=(20, 20))
        plt.suptitle('Histograms', x=0.5, y=0.92)
        plt.show()
        
    def plot_boxplot(self, data):
        """
        Plot Boxplot
        """
        data.plot(kind='box', subplots=True, layout=(4, 4), figsize=(20, 20))
        plt.suptitle('Boxplot', x=0.5, y=0.92)
        plt.show()
        
    def plot_scatter(self, data):
        """
        Plot Scatter
        """
        pd.plotting.scatter_matrix(data, figsize=(20, 20))
        plt.suptitle('Scatter Matrix', x=0.5, y=0.92)
        plt.show()
        
    def plot_countplot(self, data, column):
        """
        Plot Countplot
        """
        sns.countplot(x=column, data=data)
        plt.title(f'Countplot of {column}')
        plt.show()

    def plot_pairplot(self, data):
        """
        Plot Pairplot
        """
        sns.pairplot(data, diag_kind='kde')
        plt.title('Pairplot')
        plt.show()
    
    def plot_heatmap(self, data):
        """
        Plot Heatmap
        """
        sns.heatmap(data.corr(), annot=True)
        plt.title('Heatmap')
        plt.show()
        
    def plot_barplot(self, data, x, y):
        """
        Plot Barplot
        """
        sns.barplot(x=x, y=y, data=data)
        plt.title('Barplot')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot Confusion Matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, cmap='coolwarm')
        plt.title('Confusion Matrix')
        plt.show()
        
    def plot_roc_curve(self, y_true, y_pred):
        """
        Plot ROC Curve
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()
        
    def plot_auc_score(self, y_true, y_pred):
        """
        Plot AUC Score
        """
        auc = roc_auc_score(y_true, y_pred)
        print(f'AUC: {auc}')
        return auc
    
    
    def plot_lime(self, model, X_train, X_test, y_train, y_test):
        """
        Plot LIME
        """
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=['0', '1'], discretize_continuous=True)
        exp = explainer.explain_instance(X_test.values[0], model.predict, num_features=len(X_train.columns))
        exp.show_in_notebook(show_table=True)
        
    def plot_feature_importance(self, model, X_train):
        """
        Plot Feature Importance
        """
        importance = model.feature_importances_
        for i, v in enumerate(importance):
            print(f'Feature: {X_train.columns[i]}, Score: {v}')
        plt.bar([x for x in range(len(importance))], importance)
        plt.xticks(range(len(importance)), X_train.columns, rotation=90)
        plt.show()


if __name__ == "__main__":
    # Sample data
    data = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    target_column = np.random.choice([0, 1], size=100)
    
    # Convert target_column to string to use as hue
    target_column_str = target_column.astype(str)
    
    # Creating an instance of Visualization
    vis = Visualization()
    
    # Example of using different visualization methods
    vis.plot_correlation(data)
    vis.plot_histogram(data)
    vis.plot_boxplot(data)
    vis.plot_scatter(data)
    vis.plot_countplot(data, 'A')
    vis.plot_heatmap(data)
    vis.plot_barplot(data, 'A', 'B')
    vis.plot_pairplot(data)
