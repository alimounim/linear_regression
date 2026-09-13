from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_iris_split(test_size=0.1, random_state=42 ):
    """Load the Iris dataset and split it into training and testing sets.

    Returns
    -------
    X_train: numpy.ndarray
        The training input data.
    X_test: numpy.ndarray
        The testing input data.
    y_train: numpy.ndarray
        The training target data.
    y_test: numpy.ndarray
        The testing target data.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

    