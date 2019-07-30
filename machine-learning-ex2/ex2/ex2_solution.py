import csv
import pandas as pd
import  numpy as np
from math import  e
import scipy.optimize as op
from sklearn.metrics import  accuracy_score
import  matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# X=0;
# y=0;
# lemda=1
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

# for ex1
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

# for ex2
def costFunctionReg(theta, X, y, lemda):
    m = y.shape[0]
    J = 0;
    J = ((-y * np.log(sigmoid(theta, X))) - (1 - y) * np.log(1 - sigmoid(theta, X)));
    J = (np.sum(J)) / m ;
    a=np.dot(theta,theta);
    a=a*(lemda/(2*m));
    J=J+a;
    print("J :",J)
    return J,gradientReg(theta,X,y,lemda);



def s(z):
    g = 1.0 / (1.0 + (e ** (-z)))
    return g;
#for ex1
def gradient(theta,X,y):
    m , n = X.shape
    theta = theta.reshape((n,1));
    y = y.reshape((m,1))
    sigmoid_x_theta = s(X.dot(theta));
    grad = ((X.T).dot(sigmoid_x_theta-y))/m;
    return grad.flatten();
# for ex2
def gradientReg(theta,X,y,lemda):
    m, n = X.shape
    grad = np.matmul(X.T, sigmoid(theta, X) - y)
    grad = grad / m;
    t=theta*(lemda/m);
    t[0]=0;
    grad+theta;
    return grad
# for ex2
def mapFeatures(X,degree):
    poly = PolynomialFeatures(degree=degree)
    return (poly.fit_transform(X))


#
# def decoratedCostFUnc(theta):
#     return costFunctionReg(theta,X,y,lemda)


def gradientDescent(X, y, theta, alpha, num_iters, Lambda):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        cost, grad = costFunctionReg(theta, X, y, Lambda)
        theta = theta - (alpha * grad)
        J_history.append(cost)

    return theta, J_history

def mapFeaturePlot(x1,x2,degree):
    """
    take in numpy array of x1 and x2, return all polynomial terms up to the given degree
    """
    out = np.ones(1)
    for i in range(1,degree+1):
        for j in range(i+1):
            terms= (x1**(i-j) * x2**j)
            out= np.hstack((out,terms))
    return out


def ex1():
    data = pd.read_csv('ex2data1.txt', header=None)
    y = data.iloc[:, 2]
    X = data.iloc[:, 0:2]
    filter = y == 1
    plt.scatter(X.loc[filter, 0], X.loc[filter, 1], marker='+', label='Admitted')
    filter = y == 0
    plt.scatter(X.loc[filter, 0], X.loc[filter, 1], marker='o', label='Not Admitted')
    plt.legend()
    y = y.to_numpy()
    X = X.to_numpy()
    X=mapFeatures(X,1)


    #
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20, random_state = 112)

    X=X_train;
    y=y_train;
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
    print("Accuracy on Train : ",accuracy_score(predict(optimal_theta, X), y_train) * 100)
    print("Accuracy on Test : ", accuracy_score(predict(optimal_theta, X_test), y_test) * 100)
    x_value = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
    y_value = -(optimal_theta[0] + optimal_theta[1] * x_value) / optimal_theta[2]
    # print(x_value, y_value)
    plt.plot(x_value, y_value, "g")
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.show()


def ex2():

    data = pd.read_csv('ex2data2.txt', header=None)
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
    X = mapFeatures(X, 6)
    # plt.show()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=112)

    X = X_train;
    y = y_train;
    lemda=10
    m, n = X.shape;
    j,grad = costFunctionReg(np.zeros(n),X,y,lemda)
    print("Initial Cost : ",j)
    print("Initial gradient : ", grad[0:5])
    initial_theta = np.ones(n);
    optimal_theta, J_history = gradientDescent(X, y, initial_theta, lemda, 1000, 0.03)
    # print(optimal_theta)
    # plt.plot(J_history)

    # optimal_theta= fmin_bfgs(decoratedCostFUnc, initial_theta)
    print("Optimal Theta :", optimal_theta[0:5])
    #
    print("Accuracy on Train : ", accuracy_score(predict(optimal_theta, X_train), y_train) * 100)
    print("Accuracy on Test : ", accuracy_score(predict(optimal_theta, X_test), y_test) * 100)

    u_vals = np.linspace(-1, 1.5, 50)
    v_vals = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u_vals), len(v_vals)))
    for i in range(len(u_vals)):
        for j in range(len(v_vals)):
            z[i, j] = mapFeaturePlot(u_vals[i], v_vals[j], 6) @ optimal_theta
    plt.contour(u_vals, v_vals, z.T, 0)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend(loc=0)

    plt.show()


if __name__ == '__main__':
    ex2()
    # print(np.zeros(8))