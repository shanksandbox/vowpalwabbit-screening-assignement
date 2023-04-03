from typing import List
from sklearn.linear_model import LinearRegression

class NewTrainer:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, x: List[List[float]], y: List[float]):
        if not isinstance(x, list) or not all(isinstance(xi, list) and all(isinstance(xij, float) for xij in xi) for xi in x):
            raise TypeError("x must be a 2D list of floats")

        # Ensure that y is a 1D list of floats
        if not isinstance(y, list) or not all(isinstance(yi, float) for yi in y):
            raise TypeError("y must be a 1D list of floats")
        self.model.fit(x, y)


    def predict(self, x: List[float]) -> float:
        if len(x) == 0:
            raise ValueError("x cannot be an empty list")
        if not isinstance(x, list) or not all(isinstance(xi, float) for xi in x):
            raise TypeError("x must be a 1D list of floats")
        return self.model.predict([x])[0]