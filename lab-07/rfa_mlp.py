# this file gives a number of helpful functions for the lab!
# feel free to study at your leasure.

import numpy as np

class NeuralNetwork:
    def __init__(self, layer_dims, lr=0.03):
        self.layer_dims = layer_dims
        self.num_layers = len(self.layer_dims) - 1 
        self.lr = lr
        # dims of all W: n outputs of a layer, m inputs to a layer
        self.W = []
        for i in range(self.num_layers):
            # this is called the Xavier initialization in Deep Learning: 
            # it simply means rescaling normally distributed initial weights
            # by the 2 / #entries in the weight matrix and has prooven a generally
            # good starting point for many neural network algorithms.
            w = np.random.randn(layer_dims[i+1], layer_dims[i])
            w *= np.sqrt(2.0 / (layer_dims[i+1] + layer_dims[i]))
            self.W.append(w)

        self.b = [np.zeros((layer_dims[i+1], 1)) for i in range(self.num_layers)]

        # these will become lists in which we store linear and nonlinear activations
        # prevents passing around variables a lot.
        self.z = None
        self.h = None

    @staticmethod # static methods put methods into the scope of a class, but do not depend on dynamic data attached to an instantiation of the object.
    def _act(z):
        # we choose tanh networks here because they tend to work well with RFA
        return np.tanh(z)

    @staticmethod
    def _d_act(z):
        # the derivative of tanh.
        return 1.0 - np.tanh(z) ** 2

    def forward(self, x):
        self.h = [x] # first input
        
        self.z = [] # will be used to calculate the backpropagation signal

        for layer_idx, (W, b) in enumerate(zip(self.W, self.b)):
            z = W @ self.h[layer_idx] + b
            self.z.append(z) # store for learning

            # non-linear activation, but last layer is linear
            h = self._act(z) if layer_idx < (len(self.W) - 1) else z
            self.h.append(h) # store for learning

        # get the final network output
        return self.h[-1]

    @staticmethod
    def compute_loss(prediction, target):
        return 0.5 * np.mean( (prediction - target)**2 )


# synthetic data is often helpful for toy experiments and first implementations.
# here, we are generating a number of datapoints with an unkown map

def create_dataset(n_samples=1_000, n_features=10, n_outputs=5):                         
    """Create a dataset"""
    X = np.random.randn(n_features, n_samples)        
    W_true = np.random.randn(n_outputs, n_features) * 0.5
    y = np.tanh(W_true @ X)

    # add a bit of noise
    y += np.random.randn(n_outputs, n_samples) * 0.1

    return X, y

def train_test_split(X, y, test_ratio = 0.2):
    n = X.shape[1]
    n_test = int(n * test_ratio)
    rand_indicies = np.random.permutation(n)

    X_train = X[:, rand_indicies[n_test:]] # first n_test for testing
    X_test = X[:, rand_indicies[:n_test]] 
    y_train = y[:, rand_indicies[n_test:]] 
    y_test = y[:, rand_indicies[:n_test]] 

    return X_train, X_test, y_train, y_test

def train_network(network, 
                  X_train, 
                  y_train, 
                  X_test, 
                  y_test, 
                  epochs = 
                  500, 
                  learning_rate=0.01, 
                  batch_size=32):

    num_samples = X_train.shape[1]
    num_batches = num_samples // batch_size

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # move the train data around for each epoch
        rand_indicies = np.random.permutation(num_samples)
        X_shuffled = X_train[:, rand_indicies]
        y_shuffled = y_train[:, rand_indicies]

        epoch_loss = 0

        for i in range(num_batches):
            batch_start=  i*batch_size
            batch_end = batch_start + batch_size

            X_batch = X_shuffled[:,batch_start:batch_end]
            y_batch = y_shuffled[:,batch_start:batch_end]

            # forward:
            prediction = network.forward(X_batch)
            loss = network.compute_loss(prediction, y_batch)
            epoch_loss += loss

            network.backward(y_batch, learning_rate)


        train_losses.append(epoch_loss / num_batches)

        test_prediction = network.forward(X_test)
        test_loss = network.compute_loss(test_prediction, y_test)
        test_losses.append(test_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.4f}, Test Loss = {test_loss:.4f}")

    return train_losses, test_losses

def create_dataset_with_bad_correlations(n_samples=1000, 
                                         n_features=10, 
                                         n_outputs=5, 
                                         input_condition_number = 100,
                                         output_condition_number= 50, 
                                         noise_level=0.1):
    # Generate input covariance matrix
    A_in = np.random.randn(n_features, n_samples)
    Q_in, _ = np.linalg.qr(A_in)
    eigenvalues_in = np.logspace(0, np.log10(input_condition_number), n_features)
    eigenvalues_in = eigenvalues_in / np.mean(eigenvalues_in)  # Normalize
    S_in = Q_in @ np.diag(eigenvalues_in) @ Q_in.T
    L_in = np.linalg.cholesky(S_in) # the covariance matrix.
    
    X_uncorr = np.random.randn(n_features, n_samples)
    X = L_in @ X_uncorr # correlated inputs!

    # similarly, we can correlate outputs.
    A_out = np.random.randn(n_outputs, n_outputs)
    Q_out, _ = np.linalg.qr(A_out)
    eigenvalues_out = np.logspace(0, np.log10(output_condition_number), n_outputs)
    eigenvalues_out = eigenvalues_out / np.mean(eigenvalues_out)  # Normalize
    S_out = Q_out @ np.diag(eigenvalues_out) @ Q_out.T
    L_out = np.linalg.cholesky(S_out)
    
    # Create true mapping from inputs to outputs
    W_true = np.random.randn(n_outputs, n_features) * 0.3
    y_uncorr = np.tanh(W_true @ X)
    y = L_out @ y_uncorr # add covariances
    
    # Add correlated noise (also respecting the covariance structure)
    noise = np.random.randn(n_outputs, n_samples) * noise_level
    correlated_noise = L_out @ noise
    y += correlated_noise
    
    return X, y
