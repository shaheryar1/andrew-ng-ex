import  numpy as np
from math import  e
import scipy.optimize as op
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import multiprocessing as mp
from scipy.optimize import fmin_bfgs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from scipy.io import loadmat
def mapFeatures(X,degree):
    poly = PolynomialFeatures(degree=degree)
    return (poly.fit_transform(X))


data = loadmat('ex5data1.mat');
y = data['y']
X = data['X']
Xval = data['Xval']
yval = data['yval']
yTest = data['ytest']
XTest = data['Xtest']

def linearRegCostFunction(X,y,theta,lamda):
    m=X.shape[0]
    cost = ((theta @ X.T)-y.flatten())**2;
    cost=sum(cost)/(2*m)
    reg = np.sum(theta.flatten()[1:len(theta)]**2)*lamda/(2*m);
    cost = cost + reg;

    grad=((theta @ X.T)-y.flatten());
    grad=(grad @ X)/m;
    t=(lamda/m)*theta;
    t[0]=0;
    grad=grad+t
    return cost,grad;

def trainLinearReg(X,y,theta,num_iters,alpha,lamda):

    J_history = [0]

    for i in range(num_iters):
        cost, grad = linearRegCostFunction (X, y,theta, lamda)
        if(cost==J_history[-1]):

            break;
        theta = theta - (alpha * grad)
        J_history.append(cost)


    return theta, J_history

def learningCurve(X,y,Xval,yval,lamda):
    m=X.shape[0]
    n=X.shape[1]
    error_train=[]
    error_validation=[]

    alpha=0.1
    ini_theta=np.ones(n)
    i=2
    while i <= m:
        optimal_theta=trainLinearReg(X[:i,:],y[:i],ini_theta,3000,alpha,lamda)[0]
        train_cost=linearRegCostFunction(X[:i,:],y[:i],optimal_theta,lamda)[0]
        validation_cost=linearRegCostFunction(Xval, yval, optimal_theta, lamda)[0]
        a=error_train.append(train_cost)
        b=error_validation.append(validation_cost)
        i=i+1;



    return error_train,error_validation,optimal_theta

def polyFeatures(X,p):

    i=2
    while i<=p:
        X = np.c_[X, X[:, 0] ** i];
        i=i+1
    return X;

def featureNormalize(X):
    mu = np.mean(X,axis=0);
    X=np.add(X,-mu)
    sigma=np.std(X,axis=0);
    X=np.divide(X,sigma);
    return X,mu,sigma


def snippet1():
    # pre processing
    global X,y,yval,Xval,XTest,yTest
    X = mapFeatures(X, 1)

    lamda = 0;
    theta = np.array([1, 1])
    alpha = 0.001
    iterations = 5000
    optimal_theta, J_hist = trainLinearReg(X, y, theta, iterations, alpha, lamda)
    print(optimal_theta)
    plt.plot(X[:, 1], X[:, 1] * optimal_theta[1] + optimal_theta[0])
    plt.scatter(X[:, 1], y[:, 0])
    # validating
    Xval = mapFeatures(Xval, 1)
    print("Cost on train : ", linearRegCostFunction(X, y, optimal_theta, lamda)[0])
    print("Cost on Validation : ", linearRegCostFunction(Xval, yval, optimal_theta, lamda)[0])
    # plt.show()

def snippet2():
    global Xval,X,y,yval;
    X = mapFeatures(X, 1)
    Xval=mapFeatures(Xval,1)
    error_train,error_validation,optimal_theta=learningCurve(X,y,Xval,yval,0)
    plt.plot(error_train)
    plt.plot(error_validation)
    plt.legend("Train Loss","Validation Loss")
    plt.show()

if __name__ == '__main__':

    # mapping X train and normalizing
    X=polyFeatures(X,8)
    X,mu,sigma=featureNormalize(X)
    # Mappping  Xval and normalizing
    Xval = polyFeatures(Xval, 8)
    Xval = np.add(Xval, -mu)
    Xval = np.divide(Xval, sigma);

    # Mapping xtest and normalizing
    XTest = polyFeatures(XTest, 8)
    XTest = np.add(XTest, -mu)
    XTest = np.divide(XTest, sigma);

    # adding 1's
    X = mapFeatures(X, 1)
    Xval = mapFeatures(Xval, 1)
    XTest = mapFeatures(XTest, 1)
    lamda=3
    a,b,optimal_theta=learningCurve(X,y,Xval,yval,lamda)
    plt.xlabel("No of m")
    plt.ylabel("Error")
    plt.plot(a)
    plt.plot(b)
    plt.legend(["Train Loss","Validation Loss"])
    plt.show()

    print("Train Error : ", a[-1])
    print("Validation Error : ", b[-1])
    print("Test Error : ", linearRegCostFunction(XTest,yTest,optimal_theta,lamda)[0])


