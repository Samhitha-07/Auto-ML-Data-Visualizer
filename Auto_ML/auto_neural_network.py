# Neural Network for the Auto ML project

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

class NeuralNetwork:
    def __init__(self, input_dim, output_dim):
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=input_dim, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(output_dim, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)
        
    def predict(self, X):
        return self.model.predict(X)
    
    

        