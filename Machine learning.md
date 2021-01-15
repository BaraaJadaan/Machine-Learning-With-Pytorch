# lesson2(Regression)
* we use Gradiant Decent to minimize the Error to ive us the ideal cut between the data

## sciket-learn for linear regression
>* For linear regression there is a `LinearRegression` class in sciket-learn which has `fit` function to fit linear regression to your data: **finding the best line that fits the training data**

```python
>>> from sklearn.linear_model import LinearRegression
>>> model = LinearRegression()
>>> model.fit(x_values, y_values) #model.fit(data[['BMI']],data[['Life expectancy']])
>>> print(model.predict([12]) # return the array of predictions
```
-------------------
## polynomial Regression
* **y_hat = w1\*x^3 + w2\*x^2 +w3\*x + w4** 
```python
>>> from sklearn.preprocessing import PolynomialFeatures
```
-------------------
## Regularization
* we do it to overcome the overfitting issue
* **overfitting** is an issue where you have alot of diffrent features but a little training data that your line try to fit as much data as possiple so we need remove some features

* Performing L2 regularization has the following effect on a model :

   * Encourages weight values toward 0 (but not exactly 0)

   * Encourages the mean of the weights toward 0, with a    normal (bell-shaped or Gaussian) distribution.
   Increasing the lambda value strengthens the regularization effect

* When choosing a **lambda** value, the goal is to strike the right balance between simplicity and training-data fit:

   *  If your lambda value is too high, your model will be simple, but you run the risk of underfitting your data. Your model won't learn enough about the training data to make useful predictions.

   * If your lambda value is too low, your model will be more complex, and you run the risk of overfitting your data. Your model will learn too much about the particularities of the training data, and won't be able to generalize to new data.

* **L1 Regularization(lasso):**
>we add the absolute coefficents to the error(|W|)
* **L2 Regularization:**
>we add the square of the coefficents to the error(W^2)

L1 Regularization                                 |            L2 Regularization
---------------------------------------------     |  --------------------------------------
computationally inefficient(unless date is sparse)|         computationally efficient
sparse output                                     |         non-sparse output
feature selection                                 |         no feature selection

```python
>>> from sklearn.linear_model import Lasso
>>> lasso_reg = Lasso()
>>> lasso_reg.fit(X, y)
#then-> we Retrieve and print out the coefficients from the regression model.
>>> reg_coef = lasso_reg.coef_          #[coef_] output the slope 
```

## Feature Scaling
* its a trick to make gradiant descent run much faster in a fewer iterations by transforming your data into a common range of values

* we want to get every feature into approximately a [-1<=xi<=1]-> around zero

* There are two common scalings:

  * 1. Standardizing :
    * Standardizing is completed by taking each value of your column, subtracting the mean of the column, and then dividing by the standard deviation of the column.
        ```python
         df["height_standard"] = (df["height"] - df["height"].mean()) / df["height"].std()
        ```
    * This will create a new "standardized" column where each value is a comparison to the mean of the column and a new, standardized value can be interpreted as the number of standard deviations the original height was from the mean

  * 2. Normalizing :
    * data are scaled between 0 and 1
    ```python
    df["height_normal"] = (df["height"] - df["height"].min()) /(df["height"].max() - df['height'].min())
    ```
* we use Feature Scaling When :
    1. your algorithm uses a distance-based metric to predict.
    2. When you incorporate regularization. 

```python
# TODO: Add import statements
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Assign the data to predictor and outcome variables
# TODO: Load the data
train_data = pd.read_csv('data.csv', header = None)
X = train_data.iloc[:,:-1]#choosing from first to before last row
y = train_data.iloc[:,-1]#choosing last row

# TODO: Create the standardization scaling object.
scaler = StandardScaler()

# TODO: Fit the standardization parameters and scale the data.
X_scaled = scaler.fit_transform(X)

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
lasso_reg.fit(X_scaled, y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)
```
-------------------
# lesson 3 (Perceptron Algorithm)
## classification
* the boundary line\plane :
 has the equasion{ W*X + b = 0 }:

W : array of weights

X : array of inputs

b : the bias is an error from erroneous
 assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting). Bias is the accuracy of our predictions. A high bias means the prediction will be inaccurate

* prediction:
y_hat {=1 if WX+b>=0  \\ =0 if WX+b<0}

* the x`s in the equasion are the input nodes and the w`s are the bridge between x`s and the main node , the main node do the equasion and make the outpt(the line), then the "step function" takes the number and if it is positive then output "1"(true) and if negative output "0"(false)

## perceptron Algorithm
1. Line : 3x1 + 4x2 - 10 = 0  (Wx + b + 0)

2. Point : (4,5)

3. Learning Rate : 0.1

*  if the negative point is in the positive area (prediction = 1) we **substract** the x,y points "and 1 for bias" mutiplyed to the learning rate from the parameters of the line equasion if the point is ***above*** the line then we make a new equasion out of our new parameters ***[wi-αxi]*** and ***[b-α]***:

Line : 2.6x1 + 3.5x2 - 10.1


* if the positive point is in the negative area (prediction = 0) we **add** the x,y points "and 1 for bias" mutiplyed to the learning rate to the parameters of the line equasion if the point is ***below*** the line then we make a new equasion out of our new parameters ***[wi+αxi]*** and ***[b+α]***

-------------------
# lesson 4(decision tree)
* it tend to overfitt alot without ensabling
* an algorithm with low bias and low varience 
## Entropy and Information Gain
[Entropy formula](Entropyformula.png)
* p1 = m/m+n
* p2 = m/m+n
>* ***Entropy = −∑ pi * log2(pi)*** 
* we calculate the entropy to have the *Information Gain* :

>* ***Information Gain = Entropy(parent) - { [m/m+n]*Entropy(chaild1) + [n/m+n]*Entropy(child2) }*****
* we pick the column with the highest information gain
* [Example of decision tree](data.jpg)
* [Solution of the Examble](CutedDataForDicisionTree.jpg)

## Hyperparameters for Decision Trees
1. [Maximum Depth :](TreeDepth.png) A tree of maximum length k can have at most 2^k leaves...`max_depth`
2. [Minimum number of samples to split...](min-samples-split.png) `min_samples_leaf`
3. [Minimum number of samples per leaf...](MinimumNumberOfSamplesPerLeaf.png) `min_samples_split`
* Ex :
```python
>>> model = DecisionTreeClassifier(max_depth = 7, min_samples_leaf = 10)
```
## Steps:
1. Training the model:
    * we split into training and testing data first:
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features(X), outcomes(y), test_size=0.25, random_state=42)
    ```
```python
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```
2. Making predictions:
```python
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
```
3. Calculating accuracies
```python
from sklearn.metrics import accuracy_score
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
```
# Lesson 5(Naive bayes)
* [Example of bayes theorem](ExampleOfBayesTheorem.jpg)
* [Formula of bayes theorem](FormulaOfBayesTheorem.jpg)
* > Naive bayes : P(A | B,C ) ∝ P(B | A).P(C | A).P(A) 
-------------------
# Lesson 6(Support vector machine)
* punish the data mean that we add the data to our classification error [Click](Error=.jpg)

* > ERROR = Classification error + Margin error

* [Perceptron Algorithm to minimize the error function](PerceptronAlgorithm.jpg)

* [Classification Error](ClassificationError.jpg)

* [Margin Error](MarginError.jpg)

* > Margin Error = |W|^2
* > Margin = 2/|W|

* [C-Parameter](C-Parameter.jpg)

* **Kernel Trick**(a kit of function that willl help us with seperating) :we use it either when we have one dimentional data or two dim data to find our perfect cut
  * One-dim : we make them more dim(for ex 2) by putting y=x^2 function and fit them up to the fun after that we separate them using a line(y = x) then we subcitute y=x in y=x^2 and we have our perfect cut 

  * two-dim : we make them more (for ex 3) dim by putting y^2+x^2=r circle function and fit them up to the fun after that we saperate them using a line(y = x) then we subcitute y=x in y=x^2 and we have our perfect cut[ Kernal Trick 1](KernalTrick1.jpg)/ [Kernal Trick 2](KernalTrick2.jpg)

* **RBF Kernal**(Radial Bases Function)( we want to built mountins above or below every point to make it easyier to separate by fiding the weights and if its negative its below and above if positive...so we plot the points in higher dim and lets say 3rd degree,we separate them py a plane has an equasion for ex: "2x-4y+1z=-1",we take every weight and draw our mountins and we draw the line which cut the data in"y=-1"and the line intercept with the mountins we draw a vertical line[ "line intercept"](lineintercept.jpg))

  * there is γ ([Gama Parameter](GamaP.jpg)): if γ is small then the mountin is wide(underfitting) and if is large the mountin is narrow(overfitting)
  -----------------

  # Lesson 7 (Ensemble"combine" Methods)
  1. Bagging
  2. Boosting

* we do it to make sure that the algorithm dont overfit in high varience algorithm like "dicision tree"

#### Introducing Randomness Into Ensembles 
Another method that is used to improve ensemble methods is to introduce randomness into high variance algorithms before they are ensembled together,There are two main ways that randomness is introduced:

1. Bootstrap the data : sampling the data with replacement and fitting your algorithm to the sampled data.

2. Subset the features : in each split of a decision tree or with each algorithm used in an ensemble, only a subset of the total possible features are used.

these are the two random components used in the next algorithm you are going to see called "Random Forests" which we take columns randomly lets say three and built a dicision tree with those,then we take another three and build and we do it again..so we take our data point(if its an app recommendation lets say a person)and fit our data

#### SVM in sklearn
```python
>>> from sklearn.svm import SVC
>>> model = SVC()
>>> model.fit(x_values, y_values)
```
1. `C`: The C parameter.
2. `kernel`: The kernel. The most common ones are 'linear' poly' and 'rbf'.
3. `degree`: If the kernel is polynomial, this is the maximum degree of the monomials in the kernel.
4. `gamma` : If the kernel is rbf, this is the gamma parameter.

-------------------
# Lesson8(Model Evaluation Metric)
* How well is my model doing?
* How do we improve it?

* so we split our data into training data and testing data so we can evaluate our maachine learning model if its fit the data perfectly with small error or not(we shall never use our testing data for training)
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)#mean that 25% of our training data will be use as testing data(if 16 training then 4 testing, that 12 training and 4 testing)
```
* [Confusion Matrix](confusion.png) 

## Accuracy
* is the answer of : out of all data how many did we classify correctly
* > Accuracy = (Correctly Classified Points)/Num of all data

```python
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_true, y_pred)
```
* note that some problems we have to but the false negative down like in medical things,and other problems we need to put the false positive down like with emails(if spam or ham),so we need to implement something to these problems and let it be:**Precision** and **Recall**..like in medical things we need high recall and in email stuff we need high precision

* **Precision**(out of the points we pridicted to be Positive how many did we classify correctly?) : we implement the accuracy formula to the first *column* of the confusion matrix
>TP/TP+FP

* **Recall**(out of the points labelled Positive how many we correctly pridict?) : we implement the accuracy formula to the first *row* of the confusion matrix
>TP/TP+FN  

### F1 Score

* so we dont want to keep two scores(precision,recall) we want one score by combining those two..

* ***F1 Score*** : is a number we got when we take the harmonic mean to precision and recall

>Arithmetic Mean = (X+Y)/2

>Harmonic Mean = (2XY)/X+Y

> F1 Score = 2 ⋅ Precision * Recall/
Precision + Recall

* F1 Score = Harmonic Mean(precision , recall)

> Fβ=(1+N^2)⋅ Precision*Recall/N^2*Precision+Recall	 = (N^2/1+N^2)*Precision + (1/1+N^2)*Recall

1. If β=0, then we get **precision**.
2. If β=∞, then we get **recall**.
3. For other values of β, if they are close to 0,we get something close to precision, if they are large numbers,then we get something close to recall,and if β=1,then we get the **harmonic mean** of precision and recall.

### ROC(Receiver Operating Chracteristic)
* 1. perfrct split : 1
  2. good split : 0.8
  3. random split : 0.5

* True positive rate :  True positive/All positives
* False positive rate :  False positive/All positives   

