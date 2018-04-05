import numpy as np

X = np.array([[0,0,-1],[0,1,-1],[1,0,-1],[1,1,-1]])

y = np.array([-1,-1,-1,1])

def perceptron_sgd(X, Y):
    w = np.zeros(len(X[0]))
    eta = 1
    epochs = 20

    for t in range(epochs):
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                w = w + eta*X[i]*Y[i]

    return w

w = perceptron_sgd(X,y)
print(w)
testdata = np.array([0,0,-1])
## Print test result
print ("TEST", testdata.dot(w)>=0    )
