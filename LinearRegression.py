import numpy as np

class LinearRegression:

    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3, learning_rate=0.01):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        # reshape y so single-output and multi-output cases are handled consistently
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # TODO: Initialize the weights and bias based on the shape of X and y.
        self.weights = np.zeros((X.shape[1], y.shape[1]))
        self.bias = np.zeros(y.shape[1])

        # TODO: Implement the training loop.

        # Split the data into training and validation sets
        X_train, y_train, X_val, y_val = self.train_val_split(X, y)

        best_val_loss = float('inf')
        best_weights = self.weights.copy()
        best_bias = self.bias.copy()
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            # for each batch in the epoch, compute gradients, then update weights and bias
            for X_batch, y_batch in self.get_batches(X_train, y_train):
                dw, db = self.gradient_descent(X_batch, y_batch)
                self.update_parameters(dw, db)  

            # after all batches are processed, compute the validation loss for early stopping
            val_loss = self.score(X_val, y_val)
            if val_loss < best_val_loss:
                best_val_loss = val_loss 
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()
                epochs_without_improvement = 0
            else:   
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # restore the best weights and bias after training
        self.weights = best_weights
        self.bias = best_bias   

            
    def train_val_split(self, X, y, val_fraction=0.1):
        """Split the data into training and validation sets.

        Parameters:
        -----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        val_fraction: float
            The fraction of the data to use for validation.
        """
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        split_index = int(X.shape[0] * val_fraction)
        val_indices = indices[:split_index]
        train_indices = indices[split_index:]
        return X[train_indices], y[train_indices], X[val_indices], y[val_indices]

    def get_batches(self, X, y):
        """Generate batches of data.

        Parameters:
        -----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        for i in range(0, X.shape[0], self.batch_size):
            yield X[i:i + self.batch_size], y[i:i + self.batch_size]

    def gradient_descent(self, X_batch, y_batch):
        """Perform one step of gradient descent.

        Parameters:
        -----------
        X_batch: numpy.ndarray
            The input data for the batch.
        y_batch: numpy.ndarray
            The target data for the batch.
        """
        # TODO: Implement the gradient descent update.
        # compute and return dw, db for one batch of data
        m = X_batch.shape[0]
        prediction = self.predict(X_batch)
        dw = (2/m) * np.dot(X_batch.T, (prediction - y_batch)) + (self.regularization/m) * self.weights
        db = (2/m) * np.sum(prediction - y_batch, axis=0)
        return dw, db   
    

    def update_parameters(self, dw, db):
        """Update the weights and bias using the gradients.

        Parameters:
        -----------
        dw: numpy.ndarray
            The gradient of the weights.
        db: numpy.ndarray
            The gradient of the bias.
        """ 
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # TODO: Implement the prediction function.
        # target = Features * weights + bias
        prediction = np.dot(X, self.weights) + self.bias
        return prediction

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        prediction = self.predict(X)
        mse = np.mean((prediction - y)**2)
        return mse