import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


class PerceptronData:
    """
    DataModule class for getting perceptron data
    """

    def __init__(self):
        self.data = pd.read_csv("perceptron_toydata-truncated.txt", sep="\t")
        self.x_train = self.data[["x1", "x2"]].values
        self.y_train = self.data["label"].values

    def plot(self):
        x_trues = self.x_train[self.y_train == 1, 0]
        y_trues = self.x_train[self.y_train == 1, 1]
        x_falses = self.x_train[self.y_train == 0, 0]
        y_falses = self.x_train[self.y_train == 0, 1]
        plt.plot(x_trues, y_trues, marker="^", linestyle="", label="trues")
        plt.plot(x_falses, y_falses, marker="v", linestyle="", label="falses")


class PerceptronFromScratch:
    """
    Module for perceptron
    """
    def __init__(self, datamodule):
        self.datamodule = datamodule
        self.num_features = len(datamodule.data.columns) - 1
        self.data = datamodule.data
        self.weights = [0 for i in range(self.num_features)]
        self.bias = 0

    def forward(self, x):
        z = self.bias
        for i, _ in enumerate(self.weights):
            z += self.weights[i] * x[i]
        if z > 0:
            return 1
        else:
            return 0

    def update(self, x, y_pred):
        pred = self.forward(x)
        error = y_pred - pred
        self.bias += error
        for i, _ in enumerate(self.weights):
            self.weights[i] += error * x[i]
        return error


class Trainer:
    def __init__(self, module, datamodule):
        self.module = module
        self.datamodule = datamodule
        self.x_train = self.datamodule.data[["x1", "x2"]].values
        self.y_train = self.datamodule.data["label"].values

    def fit(self, epochs=1000):
        for epoch in range(epochs):
            total_error = 0
            for x, y in zip(self.x_train, self.y_train):
                error = self.module.update(x, y)
                total_error += abs(error)
            print(f"finished epoch {epoch} with {total_error} error")
            if total_error == 0: # early stopping if the model is perfect.
                return self.module.weights

def compute_accuracy(model, datamodule):
    """
    Method for computing the accuracy of a model with a datamodule for
    this specific example
    """
    all_x = datamodule.data[["x1","x2"]].values
    all_y = datamodule.data["label"].values
    total = 0
    for x,y in zip(all_x, all_y):
        res = model.forward(x)
        if(res  == y ):
            total += 1
    result = total/len(all_x) 
    return result


if __name__ == "__main__":
    # Load the model, the datamodule and the training module
    d = PerceptronData()
    m = PerceptronFromScratch(d)
    t = Trainer(m, d)
    # Train the model with the training module
    acc_beginning = compute_accuracy(m,d)
    t.fit()
    acc_end = compute_accuracy(m,d)
    print(f"at the beginning before training, accuracy was {acc_beginning}")
    print(f"at the end after training, accuracy is {acc_end}")
