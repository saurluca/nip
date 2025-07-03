# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


"""
Move actiavtion function out of lienar layer
Implement Sequentiel
"""

np.random.seed(42)


class Sigmoid:
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        return self._sigmoid(x)
    
    def backward(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))
    
    def __call__(self, x):
        return self.forward(x)
    
    
class MSE:
    def __init__(self):
        self.pred = None
        self.target = None
    
    def forward(self, pred, target):
        self.pred, self.target = pred, target
        return np.mean((pred - target)**2)
        
    def backward(self):
        return np.mean(0.5 * (self.pred - self.target))
    
    def __call__(self, pred, target):
        return self.forward(pred, target)


class SGD:
    def __init__(self, model, criterion, lr=0.01, momentum=0.9):
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.momentum = 0.9 # TODO currently unused
    
    def step(self):
        # calculation of gradients
        grad = self.criterion.backward()
        self.model.backward(grad)
        # TODO currently only for one linear layer
        self.model.W -= self.model.dW * self.lr        
        self.model.b -= self.model.db * self.lr    
        
    def __call__(self):
        return self.step()
    
    
class Sequential:
    def __init__(self, layers):
        self.layers = layers
        
    def backward(self, grad):
        for layer in self.layers:
            grad = layer.backward(grad)
        return grad
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def params(self):
        for layer in self.layers:
            yield from layer.params()
    
        
class LinearLayer:
    def __init__(self, in_dim: int, out_dim: int):
        # self.activation_fn = activation_fn
        self.W = 0.1 * np.random.randn(out_dim, in_dim)
        self.b = np.zeros(out_dim)
        self.dW = 0.0
        self.db = 0.0
        self.x = None
        # self.a = None
    
    def forward(self, x):
        # print(f"x: {x}, self.W {self.W}, self.b {self.b}")
        self.x = x
        return self.W @ x + self.b
        # self.a = a
        # x = self.activation_fn(a)
        # return x
    
    def backward(self, grad):
        # print(f"shape of incoming grad {grad.shape}")
        # print(f"shape of W {self.W.shape}")
        # grad = self.activation_fn.backward(self.a) * grad
        self.dW = np.outer(grad, self.x)
        self.db = grad 
        grad = self.W.T @ grad 
        return grad

    def __call__(self, x):
        return self.forward(x)
    
    def params(self):
        return self.W, self.b


def train(train_data, model, criterion, optimiser, n_epochs=10):
    train_losses = []
    outputs = []
    for epoch in tqdm(range(n_epochs)):
        train_loss = 0.0
        outputs_epoch = []
        for X, target in train_data:
            # forward pass
            pred = model(X)
            loss = criterion(pred, target)
            train_loss += loss
            outputs_epoch.append(pred)
            
            # backward pass
            optimiser.step()          
            
            # print(f"y {target}, pred {pred}, loss {loss}")
        train_losses.append(train_loss)
        outputs.append(outputs_epoch)
    return train_losses, outputs


def plot_loss(losses):
    plt.plot(losses)
    plt.show()

                    
def plot_predictions(outputs, targets):
    plt.scatter(targets, outputs) 
    plt.show()


def main():
    # config
    n_epochs = 30
    lr = 0.1
    
    # setup dummy data
    n_samples = 200
    inputs = np.random.uniform(-1, 1, size=(n_samples, 3))

    true_w = np.array([1.5, -2.0, 0.5])
    true_b = -0.1
    targets = Sigmoid()._sigmoid(inputs @ true_w + true_b)
    
    train_data = list(zip(inputs, targets))
        
    model = LinearLayer(in_dim=3, out_dim=1, activation_fn=Sigmoid())
    criterion = MSE()
    optimiser = SGD(model, criterion, lr=lr)
    
    train_losses, outputs = train(train_data, model, criterion, optimiser, n_epochs)
    plot_loss(train_losses)
    plot_predictions(outputs[-1], targets)
    
    print(f"final loss {train_losses[0]}")
    
    # print out final model params
    print(f"true W {true_w} model w {model.W} \n true b {true_b}, model b {model.b}")
    
    
if __name__ == "main":
    main()
    
    
main()