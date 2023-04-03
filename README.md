<h1>RL Open-Source Fest 2023</h1>
<h2>Screening Assignment: </h2>

Developing a Comprehensive Testing Framework for VowpalWabbit<br>
Submitted By: Shashank Kumar <br>
Github Link: https://github.com/stellarshank <br>

<h2>Problem Statement: </h2>
Let’s say we have just implemented a new training algorithm for regression with the following interface:<br>
class NewTrainer: <br>
    ... <br>
    def train(self, x: List[List[float]], y: List[float]): <br>
        ... <br>

    def predict(self, x: List[float]) -> float: <br>
        ... <br>
        return 0 <br>

Design and write test suite for it in Python using unittest or pytest frameworks.
<br>
<img src="https://github.com/stellarshank/vowpalwabbit-screening-assignement/blob/main/pic.PNG">
<h2>Solution:</h2>
•	Install and setup Python version 3.11.2 Install pip <br>
•	Install numpy, scikit_learn and pytest <br>
•	Run CMD in project directory <br>
•	Command: python –m pytest run test_linear_regression.py <br>
•	Requiremnets are available in the requirements.txt file  <br>

To run this code:- <br>
1) Clone the repository <br>
2) Download the requirements using the "pip install -r requirements.txt" <br>
3) In your command prompt, go to the directory where these files are stored <br>
4) Type "pytest" to run the code <br>

