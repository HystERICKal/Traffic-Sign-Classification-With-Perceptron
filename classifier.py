import numpy as np
def main():
    print ("hello")
    traindata = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    trainlabel = np.array([0,0,0,1])
    w =  np.zeros(3)
    eta = 1
    for epoch in range(0,100):
        for x,y in zip(traindata,trainlabel):
            r = x.dot(w) >= 0
            hata = y-r
            w+= eta * hata * x

    ## TEST
    test = np.array([1,0,1])
    print ( test.dot(w) >= 0 )
main()