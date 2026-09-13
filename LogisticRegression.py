import numpy as np


class LogisticRegression:

    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3,
                 learning_rate=0.1, max_steps=None):
        """Multinomial (softmax) Logistic Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The L2 regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        learning_rate: float
            The gradient descent step size.
        max_steps: int or None
            Optional cap on the total number of gradient descent steps (batches
            processed), across all epochs. None means no cap.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.weights = None
        self.bias = None
        self.classes_ = None

    def fit(self, X, y, batch_size=None, regularization=None, max_epochs=None,
            patience=None, max_steps=None):
        """Fit the model using batch gradient descent on the cross-entropy loss.

        Parameters:
        -----------
        X: numpy.ndarray
            The input data, shape (n_samples, n_features).
        y: numpy.ndarray
            The class labels, shape (n_samples,). Can be any hashable labels
            (e.g. ints or strings) - they get mapped to 0..k-1 internally.
        batch_size, regularization, max_epochs, patience, max_steps:
            Optional overrides for the values set in __init__.
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if regularization is not None:
            self.regularization = regularization
        if max_epochs is not None:
            self.max_epochs = max_epochs
        if patience is not None:
            self.patience = patience
        if max_steps is not None:
            self.max_steps = max_steps

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        y_idx = np.searchsorted(self.classes_, y)
        y_onehot = np.eye(n_classes)[y_idx]

        n_features = X.shape[1]
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        self.loss_history = []

        X_train, y_train, X_val, y_val = self.train_val_split(X, y_onehot)

        best_val_loss = float('inf')
        best_weights = self.weights.copy()
        best_bias = self.bias.copy()
        epochs_without_improvement = 0

        for epoch in range(self.max_epochs):
            for X_batch, y_batch in self.get_batches(X_train, y_train):
                dw, db = self.gradient_descent(X_batch, y_batch)
                self.update_parameters(dw, db)
                self.loss_history.append(self.cross_entropy_loss(X_batch, y_batch))
                if self.max_steps is not None and len(self.loss_history) >= self.max_steps:
                    break

            if self.max_steps is not None and len(self.loss_history) >= self.max_steps:
                print(f"Reached maximum number of steps: {self.max_steps}")
                break

            val_loss = self.cross_entropy_loss(X_val, y_val)
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

        self.weights = best_weights
        self.bias = best_bias

    def train_val_split(self, X, y, val_fraction=0.1):
        """Split the data into training and validation sets."""
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        split_index = int(X.shape[0] * val_fraction)
        val_indices = indices[:split_index]
        train_indices = indices[split_index:]
        return X[train_indices], y[train_indices], X[val_indices], y[val_indices]

    def get_batches(self, X, y):
        """Generate batches of data."""
        for i in range(0, X.shape[0], self.batch_size):
            yield X[i:i + self.batch_size], y[i:i + self.batch_size]

    def softmax(self, z):
        """Numerically stable softmax over the class axis.

        Parameters
        ----------
        z: numpy.ndarray
            Logits, shape (n_samples, n_classes).
        """
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """Compute class probabilities via the softmax of the linear scores."""
        logits = np.dot(X, self.weights) + self.bias
        return self.softmax(logits)

    def gradient_descent(self, X_batch, y_batch_onehot):
        """Compute the gradients of the cross-entropy loss for one batch."""
        m = X_batch.shape[0]
        probs = self.forward(X_batch)
        error = probs - y_batch_onehot
        dw = (1 / m) * np.dot(X_batch.T, error) + (self.regularization / m) * self.weights
        db = (1 / m) * np.sum(error, axis=0)
        return dw, db

    def update_parameters(self, dw, db):
        """Update the weights and bias using the gradients."""
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def cross_entropy_loss(self, X, y_onehot):
        """Mean categorical cross-entropy loss over the given samples."""
        probs = self.forward(X)
        eps = 1e-12
        return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))

    def predict_proba(self, X):
        """Return predicted class probabilities, shape (n_samples, n_classes)."""
        return self.forward(X)

    def predict(self, X):
        """Predict the class label for each sample.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.

        Returns
        -------
        numpy.ndarray
            Predicted class labels (values from self.classes_), shape (n_samples,).
        """
        probs = self.forward(X)
        class_idx = np.argmax(probs, axis=1)
        return self.classes_[class_idx]

    def score(self, X, y):
        """Compute classification accuracy on the given data."""
        predictions = self.predict(X)
        return np.mean(predictions == np.asarray(y))

    def save(self, filename):
        """Save the model parameters to a file."""
        if not filename.endswith('.npz'):
            filename += '.npz'
        np.savez(filename, weights=self.weights, bias=self.bias, classes=self.classes_)

    def load(self, filename):
        """Load the model parameters from a file."""
        if not filename.endswith('.npz'):
            filename += '.npz'
        data = np.load(filename)
        self.weights = data['weights']
        self.bias = data['bias']
        self.classes_ = data['classes']
