import csv
import pandas as pd
import  numpy as np
from math import  e
import scipy.optimize as op
from sklearn.metrics import  accuracy_score
import  matplotlib.pyplot as plt


def predict(theta,X):
    h=np.array(sigmoid(theta,X))
    for i in range(0,h.shape[0]):
        if h[i]>0.5:
            h[i]=1
        else:
            h[i]=0;

    return h;


def plotData(X,y):
    filter=y==1
    plt.scatter(X.loc[filter,0],X.loc[filter,1],marker='+',label='Admitted')
    filter=y==0
    plt.scatter(X.loc[filter,0],X.loc[filter,1],marker='o',label='Not Admitted')
    plt.legend()
    plt.show()

def sigmoid(theta,X):
    X=X.T;
    a=np.matmul(theta,X);
    g = 1.0 / (1.0 + (e**(-a)))
    return g;


def costFunction(theta,X,y):
    m=y.shape
    J = 0;
    J= ((-y* np.log(sigmoid(theta,X)))- (1-y)*np.log(1-sigmoid(theta,X)));
    J=(np.sum(J))/m;

    grad=np.matmul(X.T,sigmoid(theta,X)-y)
    grad=grad/m;
    # print(grad)
    # also return grad
    return J[0];
def costFunctionReg(theta, X, y, lemda):
    a=1


# temporary

def s(z):
    g = 1.0 / (1.0 + (e ** (-z)))
    return g;
#
def gradient(theta,X,y):
    m , n = X.shape
    theta = theta.reshape((n,1));
    y = y.reshape((m,1))
    sigmoid_x_theta = s(X.dot(theta));
    grad = ((X.T).dot(sigmoid_x_theta-y))/m;
    return grad.flatten();


def ex1():
    data = pd.read_csv('ex2data1.txt', header=None)

    y = data.iloc[:, 2]
    X = data.iloc[:, 0:2]
    filter = y == 1
    plt.scatter(X.loc[filter, 0], X.loc[filter, 1], marker='+', label='Admitted')
    filter = y == 0
    plt.scatter(X.loc[filter, 0], X.loc[filter, 1], marker='o', label='Not Admitted')
    plt.legend()

    # plotData(X,y)
    y = y.to_numpy()
    X = X.to_numpy()
    a = np.ones(X.shape[0])
    X = np.insert(X, 0, 1, axis=1)
    j = costFunction(np.array([0, 0, 0]), X, y)

    m, n = X.shape;
    initial_theta = np.zeros(n);
    Result = op.minimize(fun=costFunction,
                         x0=initial_theta,
                         args=(X, y),
                         method='TNC',
                         jac=gradient);
    optimal_theta = Result.x;
    print("Optimal Theta :", optimal_theta)

    print(accuracy_score(predict(optimal_theta, X), y) * 100)

    x_value = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
    y_value = -(optimal_theta[0] + optimal_theta[1] * x_value) / optimal_theta[2]
    print(x_value, y_value)
    plt.plot(x_value, y_value, "g")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.show()


if __name__ == '__main__':
    ex1()