<h1>RL Open-Source Fest 2023</h1>
<h2>Screening Assignment: </h2>

Developing a Comprehensive Testing Framework for VowpalWabbit<br>
Submitted By: Shashank Kumar
Github Link: https://github.com/stellarshank
vowpalwabbit-screening-assignement

<h2>Problem Statement:</h2>
Let’s say we have just implemented a new training algorithm for regression with the following interface:<br>
class NewTrainer:
    ...
    def train(self, x: List[List[float]], y: List[float]):
        ...

    def predict(self, x: List[float]) -> float:
        ...
        return 0

Design and write test suite for it in Python using unittest or pytest frameworks.

<h2>Solution:</h2>
•	Install and setup Python version 3.11.2 Install pip
•	Install numpy, scikit_learn and pytest
•	Run CMD in project directory
•	Command: python –m pytest run test_linear_regression.py
 
Requiremnets are available in the requirements.txt file 

To run this code:- 
1) Clone the repository
2) Download the requirements using the "pip install -r requirements.txt"
3) In your command prompt, go to the directory where these files are stored
4) Type "pytest" to run the code

