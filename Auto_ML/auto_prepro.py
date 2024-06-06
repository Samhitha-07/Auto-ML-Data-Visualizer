
# ------- AUTO ML -----
#      Preprocessing
# ----------------------

# @ Author: SEM 4 GROUP 9

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd

class Preprocessing:
    
    def _init_(self):
        pass
    
    # featuring scaling
    def scale_data(self, data, method='standard'):
        data=pd.DataFrame(data)
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaling method. Choose 'standard', 'minmax', or 'robust'.")
        
        scaled_data = scaler.fit_transform(data)
        scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
        return scaled_df
    
    # PCA
    def apply_pca(self, data, n_components=2):
 
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(data)
        pca_df = pd.DataFrame(data=pca_data, columns=[f'PC{i+1}' for i in range(n_components)])
        return pca_df
    
    def select_features(self, X, y, k=5):

        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        return pd.DataFrame(data=X_new, columns=selected_features)
    
    # One hot encoding
    def encode_categorical(self, data, columns):

        encoder = OneHotEncoder(drop='first', sparse=False)
        encoded_data = pd.DataFrame(encoder.fit_transform(data[columns]), columns=encoder.get_feature_names(columns))
        return pd.concat([data.drop(columns, axis=1), encoded_data], axis=1)
    
    def impute_missing_values(self, data, strategy='mean'):

        imputer = SimpleImputer(strategy=strategy)
        imputed_data = imputer.fit_transform(data)
        return pd.DataFrame(data=imputed_data, columns=data.columns)
    
    
    def label_encode(self, data, columns):

        encoder = LabelEncoder()
        for col in columns:
            data[col] = encoder.fit_transform(data[col])
        return data
    
    def outliers_iqr(self, data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data > lower_bound) & (data < upper_bound)]
    
    def remove_outliers(self, data, columns):
        for col in columns:
            data = self.outliers_iqr(data[col])
        return data
    
    