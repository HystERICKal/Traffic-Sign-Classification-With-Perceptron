import numpy as np
def main():
    ## Create train data
    traindata = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    trainlabel = np.array([0,0,0,1])
    ## Train 
    w = train(traindata,trainlabel,100)
    ## Create test data
    testdata = np.array([1,1,1])
    ## Print test result
    print ( test(w,testdata)    )

def train(traindata,trainlabel,epochs=100):
    w =  np.zeros(len(traindata[0]))
    eta = 1
    for epoch in range(0,epochs):
        for x,y in zip(traindata,trainlabel):
            r = x.dot(w) >= 0
            hata = y-r
            w+= eta * hata * x
    return w
def test(w,testdata):
    return testdata.dot(w) >= 0 
main() 
