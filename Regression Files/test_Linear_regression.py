from Linear_regression import NewTrainer
import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Test case for valid input
def test_valid_input():
    trainer = NewTrainer()
    x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    y = [1.0, 2.0, 3.0]
    trainer.train(x, y)
    assert trainer.predict([7.0, 8.0]) == pytest.approx(4.0, 0.01)

# Test case for x not being a 2D list of floats
def test_train_x_dtype():
    trainer = NewTrainer()
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    with pytest.raises(TypeError):
        trainer.train(x, y)

# Test case for y not being a 1D list of floats
def test_train_y_dtype():
    trainer = NewTrainer()
    x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    y = [[1.0], [2.0], [3.0]]
    with pytest.raises(TypeError):
        trainer.train(x, y)

# Test case for x containing non-float values
def test_train_x_non_float():
    trainer = NewTrainer()
    x = [[1.0, 2.0], [3.0, 'four'], [5.0, 6.0]]
    y = [1.0, 2.0, 3.0]
    with pytest.raises(TypeError):
        trainer.train(x, y)

# Test case for y containing non-float values
def test_train_y_non_float():
    trainer = NewTrainer()
    x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    y = [1.0, 'two', 3.0]
    with pytest.raises(TypeError):
        trainer.train(x, y)

# Test case for x containing non-float values in predict method
def test_predict_x_non_float():
    trainer = NewTrainer()
    x = [1.0, 'two', 3.0]
    y = [1.0, 2.0, 3.0]
    trainer.train([[1.0], [2.0], [3.0]], y)
    with pytest.raises(TypeError):
        trainer.predict(x)

# Test case for singleton x
def test_predict_singleton_x():
    trainer = NewTrainer()
    x = [1.0]
    y = [1.0]
    trainer.train([[1.0]], y)
    assert trainer.predict(x) == pytest.approx(1.0, 0.01)

# Test case for empty valued x
def test_empty_x():
    trainer=NewTrainer()
    x=[]
    y=[1.0,2.0,3.0]
    with pytest.raises(ValueError):
        trainer.predict(x)

# Test case for checking shape of x_test
def test_predict_x_shape():
    trainer = NewTrainer()
    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    y = [1.0, 2.0, 3.0]
    trainer.train(x, y)
    x_test = [[1, 2], [3, 4]]
    with pytest.raises(TypeError):
        trainer.predict(x_test)

# Test case for checking NaN values (if any)
def test_predict_nan_values():
    trainer = NewTrainer()
    x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    y = [1, 2, 3]
    with pytest.raises(TypeError):
        trainer.train(x, y)
    x_test = [[1, 2, 3], [4, np.nan, 6], [7, 8, 9]]
    with pytest.raises(TypeError):
        trainer.predict(x_test)
    
# Test case for checking NaN values (if any)
def test_predict_float_values():
    trainer = NewTrainer()
    x_train = [[0.2, 0.4, 0.6], [0.1, 0.3, 0.5], [0.4, 0.2, 0.8], [0.3, 0.7, 0.1], [0.6, 0.5, 0.9]]
    y_train = [0.5, 0.3, 0.8, 0.2, 0.9]
    trainer.train(x_train, y_train)
    x_test = [0.8, 0.1, 0.7]
    y_predicted = trainer.predict(x_test)
    y_expected = [1.096]
    assert np.allclose(y_predicted, y_expected)


#Now, let's check if by using a dataset, are we getting the correct outputs or not 
housing = fetch_california_housing()
X, y =  housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def test_new_trainer():
    model = NewTrainer() 
    model.train(X_train.tolist(), y_train.tolist())
    y_pred = [model.predict(x) for x in X_test.tolist()]
    mse = mean_squared_error(y_test, y_pred)
    print (mse) 
    assert mse <= 30