import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt




class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)



@d2l.add_to_class(LinearRegression)  
def forward(self, X):
    return self.net(X) # 

@d2l.add_to_class(LinearRegression)  
def loss(self, y_hat, y):
    fn = nn.MSELoss() # Default loss
    return fn(y_hat, y)

@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)


model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=100)
trainer.fit(model, data)

plt.show()
